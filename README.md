#  Urbanize: Conditioned Urban Scene Synthesis

Perceptions of urban environments, such as how safe, wealthy, or lively a place appearsm, play an
important role in city planning, social research, and understanding human visual bias. Prior work
such as Deep Learning the City (Dubey et al., 2016) has taken major steps toward quantifying these
perceptions by crowdsourcing large-scale pairwise ratings of Google Street View images. Their dataset
captures diverse neighborhoods around the world and includes human judgments on attributes like
safety, wealth, and beauty. This provides a unique opportunity to study how visual cues shape
our interpretation of the built environment.

---

## Repository Structure

- `data/` — stores all raw and preprocessed data files.
- `figures/` — stores all visualizations and grids.
- `report/` — interrim and final project reports describing problem motivation, pipeline, methodology, and results.
- `results/` — trained models and outputs, comparisons, and artifacts.
- `src/` — python scripts used to build full pipeline.

---

## Setup

Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirementsMacOS.txt
```

## Key Packages

See requirements.txt for the full environment configuration. Core libraries include:

- `tensorflow`
- `numpy`
- `pandas`
