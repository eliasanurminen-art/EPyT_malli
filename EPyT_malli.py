"""
EPyT_malli – vertailumalli Sykli-Analysoijalle (EPANET / EPyT-Flow)
===================================================================

Ohjelma lukee Puun_Istutus-ohjelman (PI) tallentaman verkon (network_data.npz),
rakentaa siitä EPANET-mallin ja laskee virtaukset ja painekorkeudet
EPyT-Flow-kirjastolla. Tuloksia voi verrata Kandi4.0:n (Sykli-Analysoija, SA)
tuloksiin.

Laskentaoletukset vastaavat Kandi4.0:aa:
  * painehäviö Darcy–Weisbachilla (EPANET laskee kitkakertoimen Colebrook–Whitella)
  * paikallishäviökerroin xi = 1,5 jokaisessa putkessa
  * vesi 80 °C: viskositeetti annetaan EPANETille suhteessa 20 °C veteen
  * kuluttajasolmun kulutus 0,0006 m3/s, 0-solmu on syöttö (EPANETissa "reservoir")
  * korkeuseroja ei ole (kaikkien solmujen korkeus 0 m)

Käyttö:
  1) Aja Puun_istutus1.0.py, joka luo network_data.npz-tiedoston.
  2) Aja tämä ohjelma. Ohjelma avaa valikon, jossa voit
       - tarkastella ja muokata alkutietoja (kulutukset, putket, syöttöpaine, lämpötila)
       - ajaa EPANET-simuloinnin
       - verrata tuloksia Kandi4.0:aan
       - tallentaa muokatut alkutiedot uuteen npz-tiedostoon
     Jos INTERAKTIIVINEN = True, ohjelma ajaa simuloinnin ja vertailun suoraan.

Asennus: pip install epyt-flow
"""

from __future__ import annotations

import contextlib
import warnings
import importlib.util
import io
import os
import time

import numpy as np

# ============================================================================
# PARAMETRIT (alkuarvot, joita voi muuttaa myös valikosta)
# ============================================================================
TIEDOSTO = "network_data.npz"     # PI:n tallentama verkko
INP_TIEDOSTO = "EPyT_malli.inp"   # tähän kirjoitetaan EPANET-malli
Q_KULUTUS = 0.0006                # m3/s, yhden kuluttajan kulutus
T_VESI_C = 80.0                   # °C
P_JUURI = 1.6e6                   # Pa, 0-solmun (syötön) paine
XI_PAIKALLIS = 1.5                # paikallishäviökerroin
G = 9.81                          # m/s2
INTERAKTIIVINEN = True            # False = aja simulointi ja vertailu suoraan
VERTAA_KANDI4 = True              # verrataanko tuloksia Kandi4.0:aan
KANDI4_TIEDOSTO = "Kandi4.0.py"
VERTAILULINKIT = [1, 3, 5, 7]     # linkit, joiden tulokset tulostetaan aina (samat kuin Kandi4.0:ssa)

# EPANETin viskositeetin perusarvo (suhteellinen viskositeetti 1,0 = 1,1e-5 ft2/s)
NU_EPANET_PERUS = 1.1e-5 * 0.3048 ** 2   # m2/s


# ============================================================================
# 1) ALKUTIEDOT: LUKU, TULOSTUS, MUOKKAUS JA TALLENNUS
# ============================================================================
def veden_ominaisuudet(T_C: float, P: float = P_JUURI) -> tuple[float, float]:
    """Palauttaa veden tiheyden [kg/m3] ja kinemaattisen viskositeetin [m2/s]
    annetussa lämpötilassa. Käyttää CoolPropia kuten Kandi4.0."""
    from CoolProp.CoolProp import PropsSI
    T = T_C + 273.15
    rho = PropsSI("D", "T", T, "P", P, "Water")
    mu = PropsSI("V", "T", T, "P", P, "Water")
    return rho, mu / rho


def lue_alkutiedot(tiedosto: str = TIEDOSTO) -> dict:
    """Lukee PI:n npz-tiedoston ja kokoaa kaikki mallin alkutiedot yhteen
    sanakirjaan. Kulutus tallennetaan solmukohtaisena vektorina, jotta
    yksittäisen solmun kulutusta voi muuttaa."""
    data = np.load(tiedosto)
    edges = data["edges"].astype(int)
    ef = data["edge_features"]
    n_solmut = int(edges.max()) + 1

    if "kuluttajat" in data.files:
        kuluttajat = data["kuluttajat"].astype(int)
    else:   # vanha tiedostomuoto: kuluttajat = solmut, joista ei lähde linkkejä
        lahtee = np.zeros(n_solmut, dtype=bool)
        lahtee[edges[:, 0]] = True
        kuluttajat = np.where(~lahtee)[0]
        kuluttajat = kuluttajat[kuluttajat != 0]

    if "sml_indeksit" in data.files:
        sml = data["sml_indeksit"].astype(int)
    else:   # päätellään: linkit, jotka eivät kuulu leveyshaun puuhun
        sml = _etsi_sml(edges, n_solmut)

    if "kulutus" in data.files:          # tämän ohjelman tallentama muokattu kulutus
        kulutus = data["kulutus"].astype(float)
    else:
        kulutus = np.zeros(n_solmut)
        kulutus[kuluttajat] = Q_KULUTUS

    return {
        "tiedosto": tiedosto,
        "edges": edges,
        "L": ef[:, 0].copy(),
        "D": ef[:, 1].copy(),
        "eps": ef[:, 3].copy(),
        "n_solmut": n_solmut,
        "kuluttajat": kuluttajat,
        "sml": sml,
        "kulutus": kulutus,          # m3/s solmuittain
        "P_juuri": P_JUURI,
        "T_C": T_VESI_C,
        "xi": XI_PAIKALLIS,
        "muokattu": False,
    }


def _etsi_sml(edges: np.ndarray, n_solmut: int) -> np.ndarray:
    """Leveyshaku 0-solmusta: linkki, joka johtaa jo käsiteltyyn solmuun, on SML."""
    from collections import deque
    ulos = [[] for _ in range(n_solmut)]
    for i, (u, v) in enumerate(edges):
        ulos[u].append(i)
    kayty = np.zeros(n_solmut, dtype=bool); kayty[0] = True
    jono, sml = deque([0]), []
    while jono:
        u = jono.popleft()
        for i in ulos[u]:
            v = edges[i, 1]
            if kayty[v]: sml.append(i)
            else: kayty[v] = True; jono.append(v)
    return np.array(sml, dtype=int)


def tulosta_alkutiedot(a: dict, kaikki_linkit: bool = False):
    """Tulostaa verkon koon, kulutukset ja putkitiedot (oletuksena vain SML:t
    ja muutama esimerkkiputki, kaikki_linkit=True tulostaa kaikki)."""
    rho, nu = veden_ominaisuudet(a["T_C"], a["P_juuri"])
    print("\n" + "=" * 72)
    print(f"ALKUTIEDOT ({a['tiedosto']}{', MUOKATTU' if a['muokattu'] else ''})")
    print("=" * 72)
    print(f"Solmuja {a['n_solmut']}, linkkejä {len(a['edges'])}, kuluttajia {len(a['kuluttajat'])}, "
          f"SML:iä {len(a['sml'])}")
    print(f"Syöttö 0-solmussa: P = {a['P_juuri'] / 1e6:.3f} MPa  (H = {a['P_juuri'] / (rho * G):.2f} m)")
    print(f"Vesi {a['T_C']:.1f} °C: rho = {rho:.1f} kg/m3, nu = {nu:.3e} m2/s  "
          f"(EPANETin suhteellinen viskositeetti {nu / NU_EPANET_PERUS:.4f})")
    print(f"Paikallishäviökerroin xi = {a['xi']}")
    kul = a["kulutus"]
    print(f"Kokonaiskulutus {kul.sum():.5f} m3/s; kulutusta {np.count_nonzero(kul)} solmussa "
          f"(min {kul[kul > 0].min():.5f}, max {kul.max():.5f} m3/s)" if kul.any() else "Ei kulutusta!")
    print(f"\n{'linkki':>6} {'u->v':>9} {'L [m]':>9} {'D [mm]':>8} {'eps [mm]':>9}")
    nayta = range(len(a["edges"])) if kaikki_linkit else sorted(set(list(range(min(5, len(a['edges'])))) + list(a["sml"])))
    for i in nayta:
        u, v = a["edges"][i]
        merkki = "  <- SML" if i in a["sml"] else ""
        print(f"{i:6d} {f'{u}->{v}':>9} {a['L'][i]:9.1f} {a['D'][i] * 1000:8.2f} {a['eps'][i] * 1000:9.4f}{merkki}")
    if not kaikki_linkit:
        print("   ... (kaikki linkit: valikon kohta 2)")


def muokkaa_kulutusta(a: dict, solmu: int | None, q: float):
    """Asettaa kulutuksen q [m3/s] yhdelle solmulle tai, jos solmu=None,
    kaikille kuluttajasolmuille. 0-solmun kulutusta ei voi asettaa."""
    if solmu is None:
        a["kulutus"][a["kuluttajat"]] = q
        print(f"Kaikkien {len(a['kuluttajat'])} kuluttajan kulutus = {q} m3/s")
    else:
        if solmu == 0 or not 0 <= solmu < a["n_solmut"]:
            print("Virheellinen solmu (0-solmu on syöttö)."); return
        a["kulutus"][solmu] = q
        print(f"Solmun {solmu} kulutus = {q} m3/s")
    a["muokattu"] = True


def muokkaa_putkea(a: dict, linkki: int, L: float | None = None, D: float | None = None,
                   eps: float | None = None):
    """Muuttaa yhden putken pituutta [m], halkaisijaa [m] tai karheutta [m].
    Arvo None jättää ominaisuuden ennalleen."""
    if not 0 <= linkki < len(a["edges"]):
        print("Virheellinen linkki."); return
    for nimi, arvo in (("L", L), ("D", D), ("eps", eps)):
        if arvo is not None:
            a[nimi][linkki] = arvo
    print(f"Linkki {linkki}: L = {a['L'][linkki]:.1f} m, D = {a['D'][linkki] * 1000:.2f} mm, "
          f"eps = {a['eps'][linkki] * 1000:.4f} mm")
    a["muokattu"] = True


def tallenna_alkutiedot(a: dict, tiedosto: str):
    """Tallentaa (muokatut) alkutiedot PI:n kanssa yhteensopivaan npz-muotoon.
    Huom: Kandi4.0 olettaa kaikille kuluttajille saman kulutuksen."""
    edges = a["edges"]
    A = np.pi * a["D"] ** 2 / 4
    ef = np.column_stack((a["L"], a["D"], A, a["eps"], edges[:, 0], edges[:, 1]))
    A12 = np.zeros((len(edges), a["n_solmut"]), dtype=np.int8)
    A12[np.arange(len(edges)), edges[:, 0]] = -1
    A12[np.arange(len(edges)), edges[:, 1]] = 1
    kuluttajat = np.where(a["kulutus"] > 0)[0]
    np.savez(tiedosto, edges=edges, edge_features=ef, adjacency=A12,
             kuluttajat=kuluttajat, sml_indeksit=a["sml"], kulutus=a["kulutus"])
    print(f"Tallennettu {tiedosto}")


# ============================================================================
# 2) EPANET-MALLIN RAKENTAMINEN JA SIMULOINTI
# ============================================================================
def solmu_id(i: int) -> str:
    """EPANET-tunnus solmulle i (0-solmu on syöttö, 'R0')."""
    return "R0" if i == 0 else f"N{i}"


def putki_id(i: int) -> str:
    """EPANET-tunnus linkille i."""
    return f"P{i}"


def kirjoita_inp(a: dict, polku: str = INP_TIEDOSTO) -> str:
    """Kirjoittaa alkutiedoista EPANET-syötetiedoston (.inp).
    Yksiköt: LPS (l/s), pituus m, halkaisija mm, Darcy–Weisbach-karheus mm.
    0-solmu mallinnetaan reservoir-solmuksi, jonka painekorkeus on vakio."""
    rho, nu = veden_ominaisuudet(a["T_C"], a["P_juuri"])
    H0 = a["P_juuri"] / (rho * G)
    rivit = ["[TITLE]", "Kandityon vertailumalli (EPyT_malli.py)", "", "[JUNCTIONS]",
             ";ID  Elev  Demand(LPS)"]
    for i in range(1, a["n_solmut"]):
        rivit.append(f"{solmu_id(i)}  0  {a['kulutus'][i] * 1000:.9f}")
    rivit += ["", "[RESERVOIRS]", ";ID  Head", f"R0  {H0:.6f}", "",
              "[PIPES]", ";ID  Node1  Node2  Length  Diameter(mm)  Roughness(mm)  MinorLoss  Status"]
    for i, (u, v) in enumerate(a["edges"]):
        rivit.append(f"{putki_id(i)}  {solmu_id(u)}  {solmu_id(v)}  {a['L'][i]:.6f}  "
                     f"{a['D'][i] * 1000:.6f}  {a['eps'][i] * 1000:.8f}  {a['xi']}  Open")
    rivit += ["", "[OPTIONS]",
              "Units  LPS",
              "Headloss  D-W",
              f"Specific Gravity  {rho / 998.2:.6f}",
              f"Viscosity  {nu / NU_EPANET_PERUS:.6f}",
              "Trials  200",
              "Accuracy  0.0000001",
              "Unbalanced  Stop",
              "Demand Model  DDA",
              "", "[COORDINATES]", ";Node  X  Y"] + [
              f"{solmu_id(i)}  {i:.1f}  0.0" for i in range(a["n_solmut"])] + [
              "", "[TIMES]", "Duration  0:00", "Hydraulic Timestep  1:00", "Report Timestep  1:00",
              "", "[REPORT]", "Status  No", "Summary  No", "", "[END]"]
    with open(polku, "w", encoding="utf-8") as f:
        f.write("\n".join(rivit))
    print(f"[EPyT] EPANET-malli kirjoitettu: {polku}  (H0 = {H0:.3f} m, "
          f"suhteellinen viskositeetti {nu / NU_EPANET_PERUS:.4f})")
    return polku


def aja_epyt_simulointi(a: dict, inp: str) -> dict:
    """Ajaa pysyvän tilan hydraulisen simuloinnin EPyT-Flow'n ScenarioSimulatorilla.
    Kaikkiin putkiin asetetaan virtausanturi ja kaikkiin solmuihin paineanturi.
    Palauttaa tilavuusvirrat [m3/s], painekorkeudet [m] ja painehäviöt [m]."""
    from epyt_flow.simulation import ScenarioSimulator
    warnings.filterwarnings("ignore", message=".*negative pressures.*")   # tulostetaan itse alla

    t_alku = time.perf_counter()               # mallin lataus + simulointi
    with ScenarioSimulator(f_inp_in=inp) as sim:
        linkit = [putki_id(i) for i in range(len(a["edges"]))]
        solmut = [solmu_id(i) for i in range(a["n_solmut"])]
        sim.set_flow_sensors(sensor_locations=linkit)
        sim.set_pressure_sensors(sensor_locations=solmut)
        t_sim = time.perf_counter()
        scada = sim.run_hydraulic_simulation()
        aika_sim = time.perf_counter() - t_sim     # pelkkä EPANETin hydraulinen ratkaisu

        Q_lps = np.asarray(scada.get_data_flows(sensor_locations=linkit))[0]
        P_m = np.asarray(scada.get_data_pressures(sensor_locations=solmut))[0]
        virtausyksikko = scada.sensor_config.flow_unit

    # Yksikkömuunnos m3/s:ksi (EPyT-Flow palauttaa virtaukset mallin yksikössä)
    kertoimet = {"LPS": 1e-3, "CMH": 1 / 3600, "CMD": 1 / 86400, "LPM": 1e-3 / 60, "CMS": 1.0}
    yks = _yksikon_nimi(virtausyksikko)
    Q = Q_lps * kertoimet.get(yks, 1e-3)

    H = P_m.copy()                     # korkeus 0 m -> painekorkeus = paine [m]
    rho, _ = veden_ominaisuudet(a["T_C"], a["P_juuri"])
    H[0] = a["P_juuri"] / (rho * G)    # EPANET ilmoittaa syöttösolmun (reservoir) paineeksi 0
    edges = a["edges"]
    h = H[edges[:, 0]] - H[edges[:, 1]]  # linkin painehäviö nimellissuunnassa
    aika = time.perf_counter() - t_alku
    return {"Q": Q, "H": H, "h": h, "yksikko": yks, "aika": aika, "aika_sim": aika_sim}


def _yksikon_nimi(koodi) -> str:
    """Muuttaa EPyT-Flow'n virtausyksikkökoodin nimeksi (esim. 'LPS')."""
    try:
        from epyt_flow.simulation.sensor_config import flowunit_to_str
        return flowunit_to_str(koodi)
    except Exception:
        return str(koodi)


# ============================================================================
# 3) TULOSTEN TULOSTUS JA VERTAILU
# ============================================================================
def valitut_linkit(a: dict) -> list[int]:
    """Palauttaa tulostettavat linkit: ensin SML:t, sitten VERTAILULINKIT.
    Linkin indeksi on sen rivi edges-taulukossa, joten se on sama kuin Kandi4.0:ssa."""
    sml = [int(x) for x in a["sml"]]
    return sml + [x for x in VERTAILULINKIT if x not in sml]


def tulosta_tulokset(a: dict, tulos: dict):
    """Tulostaa EPANETin tulokset: ensin SML:t (virtaus, painehäviö, syklin
    sulkeutuminen), sitten yhteenvedon kaikista linkeistä."""
    Q, H, h = tulos["Q"], tulos["H"], tulos["h"]
    edges = a["edges"]
    print("\n" + "=" * 72)
    print("EPANET-TULOKSET (EPyT-Flow)")
    print("=" * 72)
    print(f"EPyT-Flow'n virtausyksikkö: {tulos['yksikko']} (muunnettu m3/s)")
    print(f"Syöttö 0-solmusta: {Q[edges[:, 0] == 0].sum():.6f} m3/s, "
          f"kulutus yhteensä {a['kulutus'].sum():.6f} m3/s")

    print("\n--- SML:T JA VERTAILULINKIT " + ", ".join(str(x) for x in VERTAILULINKIT) + " ---")
    A = np.pi * a["D"] ** 2 / 4
    print(f"{'linkki':>6} {'u->v':>9} {'Q [m3/s]':>13} {'v [m/s]':>9} {'h [m]':>11} {'H_u [m]':>10} {'H_v [m]':>10}")
    for i in valitut_linkit(a):
        if i >= len(edges):
            print(f"{i:6d}   (verkossa ei ole näin montaa linkkiä)")
            continue
        u, v = edges[i]
        tunniste = "  <- SML" if i in a["sml"] else ""
        print(f"{i:6d} {f'{u}->{v}':>9} {Q[i]:13.6e} {Q[i] / A[i]:9.4f} {h[i]:11.5f} {H[u]:10.3f} {H[v]:10.3f}{tunniste}")

    print("\n--- KAIKKI LINKIT (10 suurinta virtausta ja SML:t) ---")
    print(f"{'linkki':>6} {'u->v':>9} {'Q [l/s]':>11} {'h [m]':>11}")
    nayta = sorted(set(list(np.argsort(-np.abs(Q))[:10]) + list(a["sml"])))
    for i in nayta:
        u, v = edges[i]
        print(f"{i:6d} {f'{u}->{v}':>9} {Q[i] * 1000:11.5f} {h[i]:11.5f}{'  <- SML' if i in a['sml'] else ''}")
    print(f"\nEPANET-laskenta-aika: {tulos['aika_sim'] * 1000:.1f} ms "
          f"(mallin lataus + simulointi {tulos['aika'] * 1000:.1f} ms)")
    print(f"Pienin painekorkeus: {H.min():.3f} m (solmu {int(np.argmin(H))})"
          f"{'  <- negatiivinen paine: syöttöpaine ei riitä tälle verkolle' if H.min() < 0 else ''}")


def vertaa_kandi4(a: dict, tulos: dict):
    """Ajaa Kandi4.0:n (Sykli-Analysoija) samalle verkolle ja vertaa
    tilavuusvirtoja ja painehäviöitä linkeittäin. Vertailu tehdään vain,
    jos kaikilla kuluttajilla on sama kulutus (Kandi4.0:n oletus)."""
    kul = a["kulutus"][a["kulutus"] > 0]
    if not os.path.exists(KANDI4_TIEDOSTO):
        print(f"\n[vertailu] {KANDI4_TIEDOSTO} ei löydy samasta kansiosta, vertailu ohitetaan.")
        return
    if a["muokattu"] or (kul.size and not np.allclose(kul, kul[0])):
        print("\n[vertailu] Alkutietoja on muokattu. Tallenna ne npz-tiedostoon (valikon kohta 7) "
              "ja aja Kandi4.0 sillä, jotta vertailu on reilu. Vertailu ohitetaan.")
        return

    spec = importlib.util.spec_from_file_location("kandi4", KANDI4_TIEDOSTO)
    k4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k4)
    # Asetetaan Kandi4.0:lle samat parametrit
    k4.Q_KULUTUS = float(kul[0]); k4.P_JUURI = a["P_juuri"]
    k4.T_VESI = a["T_C"] + 273.15; k4.XI_PAIKALLIS = a["xi"]
    # Kandi4.0:n funktioiden oletusarvot on lukittu latausvaiheessa, joten ne annetaan erikseen
    alk_lue, alk_arvaus, alk_vesi = k4.lue_verkko, k4.Alkuarvaus, k4.veden_ominaisuudet
    k4.lue_verkko = lambda: alk_lue(a["tiedosto"])
    k4.Alkuarvaus = lambda e, n, kul_, q_kulutus=None: alk_arvaus(e, n, kul_, float(kul[0]))
    k4.veden_ominaisuudet = lambda T=None, P=None: alk_vesi(a["T_C"] + 273.15, a["P_juuri"])
    loki = io.StringIO()
    with contextlib.redirect_stdout(loki):
        Q_features, H_sa, _ = k4.main()
    iteraatiot = [r for r in loki.getvalue().splitlines() if r.startswith("Suppeni")]

    Q_sa, h_sa = Q_features[:, 6], Q_features[:, 8]
    Q_ep, h_ep = tulos["Q"], tulos["h"]
    dQ = np.abs(Q_sa - Q_ep)
    rel = dQ / np.maximum(np.abs(Q_ep), 1e-12)

    print("\n" + "=" * 72)
    print("VERTAILU: Sykli-Analysoija (Kandi4.0) vs. EPANET (EPyT-Flow)")
    print("=" * 72)
    if iteraatiot: print("Kandi4.0: " + iteraatiot[0])
    aika_rivi = [r for r in loki.getvalue().splitlines() if r.startswith("Laskenta-aika")]
    if aika_rivi: print("Kandi4.0: " + aika_rivi[0])
    print(f"EPANET:   Laskenta-aika {tulos['aika_sim'] * 1000:.1f} ms "
          f"(mallin lataus + simulointi {tulos['aika'] * 1000:.1f} ms)\n")
    print(f"{'linkki':>6} {'u->v':>9} {'Q_SA [m3/s]':>13} {'Q_EPANET':>13} {'ero Q':>9} "
          f"{'h_SA [m]':>10} {'h_EPANET':>10} {'ero h':>8}")
    for i in valitut_linkit(a):
        if i >= len(Q_sa):
            print(f"{i:6d}   (verkossa ei ole näin montaa linkkiä)")
            continue
        u, v = a["edges"][i]
        eQ = abs(Q_sa[i] - Q_ep[i]) / max(abs(Q_ep[i]), 1e-12) * 100
        eh = abs(h_sa[i] - h_ep[i]) / max(abs(h_ep[i]), 1e-12) * 100
        print(f"{i:6d} {f'{u}->{v}':>9} {Q_sa[i]:13.6e} {Q_ep[i]:13.6e} {eQ:8.3f}% "
              f"{h_sa[i]:10.4f} {h_ep[i]:10.4f} {eh:7.3f}%{'  <- SML' if i in a['sml'] else ''}")
    print()
    print(f"Tilavuusvirrat, kaikki linkit: suurin ero {dQ.max():.3e} m3/s, "
          f"suurin suhteellinen ero {rel.max() * 100:.4f} % (linkki {int(np.argmax(rel))}), "
          f"keskimäärin {rel.mean() * 100:.4f} %")
    print(f"Painehäviöt: suurin ero {np.abs(h_sa - h_ep).max():.3e} m")
    print("Huom: EPANET laskee kitkakertoimen Swamee–Jain-likiarvolla, Kandi4.0 ratkaisee "
          "Colebrook–Whiten tarkasti.\n      Tästä syntyy noin 1 %:n ero painehäviöihin, vaikka virtaukset täsmäävät.")
    H_ero = np.abs(H_sa - tulos["H"])
    print(f"Painekorkeudet: suurin ero {np.nanmax(H_ero):.3e} m")


# ============================================================================
# 4) VALIKKO (alkutietojen muokkaustyökalu)
# ============================================================================
def _kysy_luku(teksti: str, oletus=None, tyyppi=float):
    """Kysyy luvun käyttäjältä. Tyhjä vastaus palauttaa oletusarvon."""
    vastaus = input(f"{teksti}{f' [{oletus}]' if oletus is not None else ''}: ").strip().replace(",", ".")
    if vastaus == "":
        return oletus
    try:
        return tyyppi(vastaus)
    except ValueError:
        print("Virheellinen luku."); return None


def simuloi_ja_tulosta(a: dict):
    """Kirjoittaa .inp-tiedoston, ajaa EPANETin ja tulostaa tulokset ja vertailun."""
    t0 = time.perf_counter()
    inp = kirjoita_inp(a)
    print(f"[EPyT] .inp-tiedoston kirjoitus: {(time.perf_counter() - t0) * 1000:.1f} ms")
    tulos = aja_epyt_simulointi(a, inp)
    tulosta_tulokset(a, tulos)
    if VERTAA_KANDI4:
        vertaa_kandi4(a, tulos)
    return tulos


def valikko(a: dict):
    """Tekstivalikko alkutietojen tarkasteluun, muokkaukseen ja simulointiin."""
    while True:
        print("\n----------------------- VALIKKO -----------------------")
        print(" 1  Näytä alkutiedot (yhteenveto)")
        print(" 2  Näytä kaikki putket")
        print(" 3  Muuta kaikkien kuluttajien kulutusta")
        print(" 4  Muuta yhden solmun kulutusta")
        print(" 5  Muuta putken pituutta / halkaisijaa / karheutta")
        print(" 6  Muuta syöttöpainetta, lämpötilaa tai paikallishäviökerrointa")
        print(" 7  Tallenna alkutiedot npz-tiedostoon")
        print(" 8  Lataa alkutiedot uudelleen tiedostosta")
        print(" 9  Aja EPANET-simulointi (ja vertailu Kandi4.0:aan)")
        print(" 0  Lopeta")
        valinta = input("Valinta: ").strip()

        if valinta == "1":
            tulosta_alkutiedot(a)
        elif valinta == "2":
            tulosta_alkutiedot(a, kaikki_linkit=True)
        elif valinta == "3":
            q = _kysy_luku("Kulutus [m3/s]", Q_KULUTUS)
            if q is not None: muokkaa_kulutusta(a, None, q)
        elif valinta == "4":
            s = _kysy_luku("Solmu", tyyppi=int)
            q = _kysy_luku("Kulutus [m3/s]", float(a["kulutus"][s]) if s is not None and 0 <= s < a["n_solmut"] else None)
            if s is not None and q is not None: muokkaa_kulutusta(a, s, q)
        elif valinta == "5":
            i = _kysy_luku("Linkki", tyyppi=int)
            if i is None or not 0 <= i < len(a["edges"]):
                print("Virheellinen linkki."); continue
            L = _kysy_luku("Pituus [m]", round(float(a["L"][i]), 3))
            D = _kysy_luku("Halkaisija [m]", round(float(a["D"][i]), 6))
            eps = _kysy_luku("Karheus [m]", float(a["eps"][i]))
            muokkaa_putkea(a, i, L, D, eps)
        elif valinta == "6":
            p = _kysy_luku("Syöttöpaine [Pa]", a["P_juuri"])
            t = _kysy_luku("Veden lämpötila [°C]", a["T_C"])
            xi = _kysy_luku("Paikallishäviökerroin xi", a["xi"])
            if None not in (p, t, xi):
                a["P_juuri"], a["T_C"], a["xi"] = p, t, xi
                a["muokattu"] = True
                print("Parametrit päivitetty.")
        elif valinta == "7":
            nimi = input("Tiedostonimi [network_data_muokattu.npz]: ").strip() or "network_data_muokattu.npz"
            if nimi == TIEDOSTO and input(f"Korvataanko {TIEDOSTO}? (k/e): ").lower() != "k":
                continue
            tallenna_alkutiedot(a, nimi)
        elif valinta == "8":
            nimi = input(f"Tiedosto [{a['tiedosto']}]: ").strip() or a["tiedosto"]
            a.clear(); a.update(lue_alkutiedot(nimi))
            print(f"Ladattu {nimi}")
        elif valinta == "9":
            simuloi_ja_tulosta(a)
        elif valinta == "0":
            break
        else:
            print("Tuntematon valinta.")


def main():
    alkutiedot = lue_alkutiedot(TIEDOSTO)
    tulosta_alkutiedot(alkutiedot)
    if INTERAKTIIVINEN:
        valikko(alkutiedot)
    else:
        simuloi_ja_tulosta(alkutiedot)


if __name__ == "__main__":
    main()
