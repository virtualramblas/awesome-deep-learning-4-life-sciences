# awesome-deep-learning-4-life-sciences
This repo lists a collection of resources for performing Deep Learning in [Python](https://www.python.org) for Life Sciences. Since the end of 2021 I started observing an always growing volume of academic work and Open Source initiatives related to topics such as biochemistry, genetics, molecular biology, bioinformatics, etc. Being my study background in Biomedical Engineering and Deep Learning, coming from past experiences as Software Engineer and currently working on applying ML/DL to real-life use cases in the pharma industry, these new Open Source efforts have caught my interest. That's why finally I decided to start this repository to provide researchers, developers and practitioners a single place to keep track of the latest developments in this space, with focus in particular on biotech and pharma.  
![Splash Image](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/images/dallemini_2022-6-27_15-56-36-cropped.png)  
Image generated through [DALL-E mini](https://huggingface.co/spaces/dalle-mini/dalle-mini) by prompting "A fancy protein folding".  
## 🤝 Contributing
Contributions welcome! Please submit PRs with new papers, repos, datasets, or tutorials.  
  
## :card_index: Cross-Link Matrix (Methods × Domains)

| Domain / Method          | CNNs | Transformers | GNNs | GANs | VAEs | Diffusion |
|---------------------------|------|--------------|------|------|------|-------------------|
| Proteomics & Protein Folding | – | ✅ ([AlphaFold](https://github.com/deepmind/alphafold), [Feature Reuse and Scaling](https://github.com/microsoft/protein-transfer)) | – | – | – | - |
| Drug Discovery & Design    | – | - | ✅ ([HGAN-DTI](https://github.com/Zora-LM/HGAN-DTI), [Deep Surrogate Docking](https://github.com/ryienh/graph-dock)) | - | – | ✅ ([DiffSBDD](https://github.com/arneschneuing/DiffSBDD), [PMDM](https://github.com/Layne-Huang/PMDM/tree/main)) |

Legend: ✅ indicates examples available in that category.  

:point_right: **Papers and models published before 2023 have been moved to the [Archived](docs/Archived.md) section of this repo.** :point_left:    

---

## :microscope: Proteomics & Protein Folding
To be written.  

---

## :pill: Drug Discovery & Design
To be written.  

---

## :bar_chart: Benchmarking & Datasets
* [DECIMER - Hand-drawn molecule images dataset](https://zenodo.org/record/6456306#.YravvC8w1QI) - A standardised, openly available benchmark dataset of 5088 hand-drawn depictions of diversely picked chemical structures. [[Article](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00620-9)]  
* [UniProt](https://www.uniprot.org/) - The world’s leading high-quality, comprehensive and freely accessible resource of protein sequence and functional information.  
* [UniLanguage](https://github.com/alrojo/UniLanguage) - Homology reduced UniProt, train-/valid-/testsets for language modeling.  
* [ChEMBL](https://www.ebi.ac.uk/chembl/) - A manually curated database of bioactive molecules with drug-like properties. It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs.  
* [Molecule OCR Real images Dataset](https://zenodo.org/record/5356500#.YrdYxi8w1QI) - Test dataset from paper "Image2SMILES: Transformer-based Molecular Optical Recognition Engine". It contains 296 structures: images and Functional Groups SMILES (FG-SMILES).  [[Paper](https://chemistry-europe.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/cmtd.202100069)]
* [FS-Mol](https://github.com/microsoft/FS-Mol) - A Few-Shot Learning Dataset of Molecules, containing molecular compounds with measurements of activity against a variety of protein targets. 
* [ProteinNet](https://github.com/aqlaboratory/proteinnet) - A standardized data set for Machine Learning of protein structure.  
* [SidechainNet](https://github.com/jonathanking/sidechainnet) - An all-atom protein structure dataset for Deep Learning. It is an extension of the ProteinNet dataset. [[Paper](https://arxiv.org/pdf/2010.08162v2.pdf)]  
* [DIPS](https://github.com/drorlab/DIPS) - **D**atabase of **I**nteracting **P**rotein **S**tructures. [[Paper](https://arxiv.org/abs/1807.01297)]  
* [Aggregated Views of Proteins](https://www.ebi.ac.uk/pdbe/pdbe-kb/proteins/P38129/structures) - Protein data bank in Europe knowledge base.  
* [ProtCAD](http://dunbrack2.fccc.edu/protcad/) - **Pro**tein  **C**ommon **A**ssembly **D**atabase. A comprehensive structural resource of protein complexes. [[Paper](https://www.biorxiv.org/content/10.1101/2022.08.15.503938v1)]  
* [gget](https://github.com/pachterlab/gget) - A free, Open Source command-line tool and Python package that enables efficient querying of genomic databases.  
* [ESM Atlas](https://esmatlas.com) - An open atlas of 617 million metagenomic protein structures.  
* [Progres](https://github.com/jgreener64/progres) - A Python package to perform fast search structures against pre-embedded structural databases and pre-embed datasets. [[Paper](https://www.biorxiv.org/content/10.1101/2022.11.28.518224v1)]  
* [ZINC](https://zinc15.docking.org/) - A free public resource for ligand discovery. The database contains over twenty million commercially available molecules in biologically relevant representations that may be downloaded in popular ready-to-dock formats and subsets. [[Paper](https://pubs.acs.org/doi/10.1021/ci3001277)]  
* [Papyrus](https://github.com/CDDLeiden/Papyrus-scripts) - A large-scale curated dataset aimed at bioactivity predictions. [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00672-x)]  
* [MISATO](https://github.com/t7morgen/misato-dataset/) - Machine Learning dataset of protein-ligand complexes for structure-based drug discovery. [[Paper](https://www.biorxiv.org/content/10.1101/2023.05.24.542082v2)]
* [OpenProteinSet](https://registry.opendata.aws/openfold/) - Training data for structural biology at scale. [[Paper](https://arxiv.org/abs/2308.05326)]  

---

## :hammer_and_wrench: Tools, Frameworks & Libraries
* [nglview](https://github.com/nglviewer/nglview) - A Jupyter widget to interactively view molecular structures and trajectories.  
* [Panel-Chemistry](https://github.com/marcskovmadsen/panel-chemistry) - Easy exploratory data analysis and build powerful data and viz tools within the domain of Chemistry using Python and HoloViz Panel.  
* [libmolgrid](https://github.com/gnina/libmolgrid) - A comprehensive library for fast, GPU accelerated molecular gridding for Deep Learning workflows. [[Paper](https://arxiv.org/pdf/1912.04822.pdf)]  
* [stmol](https://github.com/napoles-uach/stmol) - A component for building interactive molecular 3D visualizations within Streamlit web applications.  
* [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) - An add-on and set of pre-made nodes for Blender & Blender’s Geometry Nodes, to import, animate and manipulate molecular data.  
* [Jupyter Dock](https://github.com/AngelRuizMoreno/Jupyter_Dock) - Molecular Docking integrated in Jupyter notebooks.  
* [AugLiChem](https://github.com/BaratiLab/AugLiChem) - A data augmentation library for chemical structures. [[Paper](https://arxiv.org/abs/2111.15112)]  
* [Chemiscope](https://github.com/lab-cosmo/chemiscope) - An interactive structure/property explorer for materials and molecules. While the core project is implemented in a different programming language, it has been added to this list because it provides Python extensions that allow using it within a Jupyter or Colab notebook.  
* [ReMODE](https://github.com/micahwang/ReMode/tree/master) - A Deep Learning-based web server for target-specific drug design. [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00665-w)]
* [NoLabs](https://github.com/BasedLabs/NoLabs) - An Open Source biolab that lets you run experiments with the latest state-of-the-art models, bioinformatics tools and scalable no-code workflow engine for bio research.
* [Geom3D](https://github.com/chao1224/Geom3D) - A platform for geometric modeling on 3D structures. [[Paper](https://openreview.net/forum?id=ygXSNrIU1p)]    