"""Paper-traceable channel and dataset reference specs for channel estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperReference:
    name: str
    paper_url: str
    code_url: str
    data_summary: str


ADAFORTITRAN_REFERENCE = PaperReference(
    name="AdaFortiTran",
    paper_url="https://arxiv.org/abs/2505.09076",
    code_url="https://github.com/BerkIGuler/OFDMChannelGenerator",
    data_summary=(
        "Official MATLAB generator for 120x14 SISO 5G NR OFDM grids with "
        "TDL-A, N=3 frequency pilot spacing, pilot symbols at columns 3 and 12 "
        "(1-based), train/val/test splits totaling 144k samples."
    ),
)


AMMSE_REFERENCE = PaperReference(
    name="Attention-Aided MMSE",
    paper_url="https://arxiv.org/abs/2506.00452",
    code_url="https://github.com/TaeJun1999/Attention-aided-MMSE",
    data_summary=(
        "Public code expects external 5G NR TDL-E MATLAB files with 72x14 "
        "OFDM grids and DM-RS-derived pilot observations. The upstream repo "
        "does not ship a generator or the referenced .mat files."
    ),
)

