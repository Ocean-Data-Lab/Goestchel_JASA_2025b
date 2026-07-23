[![DOI](https://zenodo.org/badge/829659096.svg)](https://doi.org/10.5281/zenodo.21222284)

# [Automated association of fin whale calls for localization using distributed acoustic sensing](https://doi.org/10.1121/10.0044257)


This code generates the figures related to the published paper [Automated association of fin whale calls for localization using distributed acoustic sensing]((https://doi.org/10.1121/10.0044257)). It serves as a tutorial for the fin whale calls automated association pipeline described in the paper. 


```{note}
Please cite this code and its associated paper as:

- Quentin Goestchel, William S. D. Wilcock, Shima Abadi; Automated association of fin whale calls for localization using distributed acoustic sensing. J. Acoust. Soc. Am. 1 July 2026; 160 (1): 135–146. https://doi.org/10.1121/10.0044257

- Quentin Goestchel & Léa Bouffaut (2025). DAS4Whales: A Python package to analyze Distributed Acoustic Sensing (DAS) data for marine bioacoustics (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.15278387
```

## Part of the datasets for the [DCLDE 2027 conference](https://www.dclde.org/)

The annotations and detections used in this paper are part of the [localization dataset](https://www.dclde.org/) of the 2027 DCLDE conference. If you want a challenge and benchmark association and localization methods, go check it out ! 

[**10.25921/v2vh-8w16**](https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/passive_acoustic//iso/xml/DCLDE_2027_DAS-finwhale-localization.xml&view=getDataView)

## Python environment and [das4whales](https://das4whales.readthedocs.io/en/latest/src/install.html) installation
The association code relies on functions that were contributed to the [`das4whales`](https://github.com/DAS4Whales/DAS4Whales) package. The dependencies of [`das4whales`](https://github.com/DAS4Whales/DAS4Whales) should be sufficient to run the scripts related to the sections of the article. 

In command line, create a virtual environment for running the code:

```shell
uv venv
```

Activate the environment 

```shell
source .venv/bin/activate
```

Install [`das4whales`](https://github.com/DAS4Whales/DAS4Whales) and its dependencies

```shell
uv pip install das4whales
```

## Data 
Most of the data neede to run the script are available in the github repository directly. However, to run `main_section5.py`, the results of the association process for the five batches of the study are needed. To download them :
```shell
wget "https://zenodo.org/records/21220346/files/associations.zip?download=1" -O data/associations.zip
unzip -d data/ data/associations.zip && rm data/associations.zip
```

## Makefile 
All the scripts can be run using the `makefile`. Example for the section 3:

```shell
make section3
```

## Scripts description 
The scripts in this repository are related to the sections of the paper [Automated association of fin whale calls for localization using distributed acoustic sensing]() and follow its organization. They depend on functions developed in [DAS4whales](https://github.com/DAS4Whales/DAS4Whales) and show how the automated association can be run on subsets of the data. Namely:
- `main_section2.py` Plot the map and the detections of the fin whale calls
- `main_section3.py` Gives an example of the SNR-weighted KDE and the first step of the association
- `main_section3c.py` Associate calls on 60s of data and plot the result
- `main_section3d.py` Associate calls on 60s of data with different result and plot the corresponding annotations.
- `main_section5.py`. Localize calls from associations. Data from [10.5281/zenodo.21220345](https://doi.org/10.5281/zenodo.21220345) are needed.    

## DATA 

The data used in this code comes from the 2021 OOI RCA dataset:

>Wilcock, W., & Ocean Observatories Initiative. (2023). Rapid: A Community Test of Distributed Acoustic Sensing on the Ocean Observatories Initiative Regional Cabled Array [Data set]. Ocean Observatories Initiative. https://doi.org/10.58046/5J60-FJ89

## Warning: high RAM usage
The scripts are memory intensive, and at least 32GB of RAM is recommended. Otherwise, the number of channels:
