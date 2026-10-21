"""Anti-kt reconstruction with the installed FastJet (through hepyy/cppyy).

Events are clustered in a small compiled loop so that per-particle Python
calls are avoided. Results: number of inclusive jets, leading-jet kinematics
and the jet index (pT-ordered) of every particle.
"""

import numpy as np

_READY = False

_CODE = r"""
#include <fastjet/ClusterSequence.hh>
#include <vector>
#include <cmath>
namespace jqi {
// One call clusters nev events of npart particles (fixed count; padded
// particles must have e <= 0 and are skipped).
void cluster(int nev, int npart, const double* px, const double* py,
             const double* pz, const double* e, double R, int* njets,
             double* jpt, double* jeta, double* jphi, int* jn, int* member) {
  fastjet::JetDefinition jd(fastjet::antikt_algorithm, R);
  std::vector<fastjet::PseudoJet> parts;
  parts.reserve(npart);
  for (int i = 0; i < nev; ++i) {
    parts.clear();
    for (int k = 0; k < npart; ++k) {
      int j = i * npart + k;
      member[j] = -1;
      if (e[j] <= 0) continue;
      fastjet::PseudoJet p(px[j], py[j], pz[j], e[j]);
      p.set_user_index(k);
      parts.push_back(p);
    }
    fastjet::ClusterSequence cs(parts, jd);
    std::vector<fastjet::PseudoJet> jets = fastjet::sorted_by_pt(cs.inclusive_jets(0.0));
    njets[i] = (int)jets.size();
    if (jets.empty()) { jpt[i] = 0; jeta[i] = 0; jphi[i] = 0; jn[i] = 0; continue; }
    jpt[i] = jets[0].pt();
    jeta[i] = jets[0].eta();
    jphi[i] = jets[0].phi_std();
    jn[i] = (int)jets[0].constituents().size();
    for (size_t a = 0; a < jets.size(); ++a)
      for (const auto& c : jets[a].constituents())
        member[i * npart + c.user_index()] = (int)a;
  }
}
}
"""


def _ensure():
    global _READY
    if _READY:
        return
    import hepyy

    hepyy.load("fastjet")
    import cppyy

    cppyy.cppdef(_CODE)
    _READY = True


def fastjet_version():
    _ensure()
    import cppyy

    return str(cppyy.gbl.fastjet.fastjet_version_string())


def cluster_events(p4, radius=0.4):
    """Cluster events.

    Args:
        p4: [N, P, 4] array (px, py, pz, E); entries with E <= 0 are ignored.
    Returns:
        dict with njets [N], jet_pt/eta/phi [N] (leading jet), jet_nconst [N]
        and membership [N, P] (pT-ordered jet index, -1 if absent).
    """
    _ensure()
    import cppyy

    p4 = np.ascontiguousarray(p4, dtype=np.float64)
    n, npart = p4.shape[:2]
    comp = [np.ascontiguousarray(p4[..., k]).reshape(-1) for k in range(4)]
    out = {
        "njets": np.zeros(n, dtype=np.int32),
        "jet_pt": np.zeros(n),
        "jet_eta": np.zeros(n),
        "jet_phi": np.zeros(n),
        "jet_nconst": np.zeros(n, dtype=np.int32),
        "membership": np.zeros(n * npart, dtype=np.int32),
    }
    if n:
        cppyy.gbl.jqi.cluster(n, npart, *comp, float(radius), out["njets"],
                              out["jet_pt"], out["jet_eta"], out["jet_phi"],
                              out["jet_nconst"], out["membership"])
    out["membership"] = out["membership"].reshape(n, npart)
    return out


def four_vectors(eta, phi, pt):
    """Massless four-vectors (px, py, pz, E) from (eta, phi, pT)."""
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta),
                     pt * np.cosh(eta)], -1)
