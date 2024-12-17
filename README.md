# awesome-deep-learning-4-life-sciences
This repo lists a collection of resources for performing Deep Learning in [Python](https://www.python.org) for Life Sciences. Since the end of 2021 I started observing an always growing volume of academic work and Open Source initiatives related to topics such as biochemistry, genetics, molecular biology, bioinformatics, etc. Being my study background in Biomedical Engineering and Deep Learning, coming from past experiences as Software Engineer and currently working on applying ML/DL to real-life use cases in the pharma industry, these new Open Source efforts have caught my interest. That's why finally I decided to start this repository to provide researchers, developers and practitioners a single place to keep track of the latest developments in this space, with focus in particular on biotech and pharma.  
Contributions, suggestions and stars are welcome!  
![Splash Image](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/images/dallemini_2022-6-27_15-56-36-cropped.png)  
Image generated through [DALL-E mini](https://huggingface.co/spaces/dalle-mini/dalle-mini) by prompting "A fancy protein folding".  
  
  
# Table of contents
[Molecules](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#molecules)  
[Proteins](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#proteins)  
[Cheminformatics](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#cheminformatics)  
[Drug Discovery](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#drug-discovery)  
[Datasets](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#datasets)  
[Explainable AI](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#explainable-ai)  
[Other](https://github.com/virtualramblas/awesome-deep-learning-4-life-sciences/blob/main/README.md#other)  
# Molecules
* [pysmiles](https://github.com/pckroon/pysmiles) - A lightweight Python library for reading and writing SMILES strings.
* [SmilesDrawer](https://colab.research.google.com/drive/1PmJX2XFg_1ORexGTw4FkKjr779sk1zAr?usp=sharing) - A Colab notebook to draw from SMILES strings.  
* [PySMILESUtils](https://github.com/MolecularAI/pysmilesutils) - Utilities for working with SMILES based encodings of molecules for Deep Learning (PyTorch oriented).  
* [SELFIES](https://github.com/aspuru-guzik-group/selfies) - Robust representation of semantically constrained graphs, in particular for molecules in chemistry.
* [ChemProp](https://github.com/chemprop/chemprop) - Message Passing Neural Networks for molecule property prediction.  
    * [Evidential Deep Learning for Guided Molecular Property Prediction and Discovery](https://github.com/aamini/chemprop) - Fast and scalable uncertainty quantification for neural molecular property prediction, accelerated optimization, and guided virtual screening. [[Paper](https://pubs.acs.org/doi/10.1021/acscentsci.1c00546)]
* [mols2grid](https://github.com/cbouy/mols2grid) - Interactive molecule viewer for 2D structures.   
* [Image to SMILES Generator](https://github.com/syntelly/img2smiles_generator) - Code to generate datasets of pairs "image - sequence" for chemical molecules. [[Article](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cmtd.202100069)]. 
* [Auto3D](https://github.com/isayevlab/Auto3D_pkg) - Automatic generation of the low-energy 3D structures with ANI Neural Network potentials.  
* [MolDQN](https://github.com/google-research/google-research/tree/master/mol_dqn) - Optimization of molecules via Deep Reinforcement Learning. [[Paper](https://arxiv.org/pdf/1810.08678v3.pdf)]  
* [Pasithea](https://github.com/aspuru-guzik-group/Pasithea) - Deep Molecular Dreaming: Inverse Machine Learning for de-novo molecular design and interpretability with surjective representations. [[Paper](https://arxiv.org/pdf/2012.09712v1.pdf)]  
* [fragment-based-dgm](https://github.com/marcopodda/fragment-based-dgm) - A Deep Generative Model for fragment-based molecule generation. [[Paper](https://arxiv.org/pdf/2002.12826v1.pdf)]  
* [MAT](https://github.com/ardigen/MAT) - **M**olecule **A**ttention **T**ransformer for molecular prediction tasks. [[Paper](https://arxiv.org/pdf/2002.08264v1.pdf)]. 
* [Specklit](https://github.com/avrabyt/st-speckmol) - A Streamlit Component for creating [Speck](https://github.com/wwwtyro/speck) molecular structures within a Streamlit Web app.  
* [molcloud](https://github.com/whitead/molcloud) - A package to draw molecules in a big canvas packed together.  
* [Img2Mol](https://github.com/bayer-science-for-a-better-life/Img2Mol) - Inferring molecules from pictures.  
* [GLAMOUR](https://github.com/learningmatter-mit/GLAMOUR) - Chemistry-informed Macromolecule Graph Representation for Similarity Computation, Unsupervised and Supervised Learning. [[Paper](https://iopscience.iop.org/article/10.1088/2632-2153/ac545e)]  
* [MOSES](https://github.com/molecularsets/moses) - **Mo**lecular **Se**t**s**: a benchmarking platform for Molecular Generation Models. [[Paper]](https://arxiv.org/abs/1811.12823)  
* [Tartarus](https://github.com/aspuru-guzik-group/Tartarus) - A benchmarking platform for realistic and practical inverse molecular design. [[Paper](https://arxiv.org/abs/2209.12487)]  
* [Transformer-M](https://github.com/lsj2408/Transformer-M) - One Transformer that can understand both 2D & 3D molecular data. [[Paper](https://arxiv.org/abs/2210.01765)]  
* [GraphINVENT](https://github.com/MolecularAI/GraphINVENT) - A platform for graph-based molecular generation using graph neural networks.  
* [SynNet](https://github.com/wenhao-gao/SynNet) - An amortized approach to synthetic tree generation using neural networks. This model can serve as both a synthesis planning tool and as a tool for synthesizable molecular design.  [[Paper](https://arxiv.org/abs/2110.06389)]  
* [SPIB](https://github.com/tiwarylab/State-Predictive-Information-Bottleneck) - SPIB (**S**tate **P**redictive **I**nformation **B**ottleneck) is a Deep Learning-based framework that learns the reaction coordinates from high dimensional molecular simulation trajectories.  [[Paper](https://aip.scitation.org/doi/abs/10.1063/5.0038198)]  
* [MolT5](https://github.com/blender-nlp/MolT5) - A self-supervised learning framework for pretraining models on a vast amount of unlabeled natural language text and molecule strings. [[Paper](https://arxiv.org/abs/2204.11817)]  
* [DIONYSUS](https://github.com/aspuru-guzik-group/dionysus) - An extensive study of the calibration and generalizability of probabilistic Machine Learning models on small chemical datasets. [[Paper](https://arxiv.org/abs/2212.01574)]  
* [NVIDIA-PCQM4Mv2](https://github.com/jfpuget/NVIDIA-PCQM4Mv2) - Heterogenous ensemble of models for Molecular Property Prediction. [[Paper](https://arxiv.org/abs/2211.11035)]  
* [JAEGER](https://github.com/Novartis/JAEGER) - JT-VAE Generative Modeling (JAEGER) is a deep generative approach for small-molecule design. It is based on the Junction-Tree Variational Auto-Encoder (JT-VAE) method. [[JT-VAE paper](https://arxiv.org/abs/1802.04364)]  
* [Chem Faiss](https://github.com/ritabratamaiti/Chem-Faiss) - Vector similarity search functionality from Faiss, in conjunction with chemical fingerprinting to build a scalable similarity search architecture for compounds/molecules.  
* [DECIMER](https://github.com/Kohulan/DECIMER-Image-to-SMILES) - Deep lEarning for Chemical ImagE Recognition (DECIMER): it translates a bitmap image of a molecule into a SMILES. [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00469-w)]  
  * [DECIMER Image Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer) - The DECIMER (Deep lEarning for Chemical ImagE Recognition) 2.1 project.  
* [STOUT](https://github.com/Kohulan/Smiles-TO-iUpac-Translator) - Transformer based SMILES to IUPAC Translator.  
* [MoLFormer](https://github.com/IBM/molformer) - A large-scale chemical language model designed with the intention of learning a model trained on small molecules which are represented as SMILES strings. [[Paper](https://www.nature.com/articles/s42256-022-00580-7)]  
* [Mol-CycleGAN](https://github.com/ardigen/mol-cycle-gan) - A generative model for molecular optimization. [[Paper](https://www.researchgate.net/publication/338466527_Mol-CycleGAN_A_generative_model_for_molecular_optimization)]  
* [CLAMP](https://github.com/ml-jku/clamp) - CLAMP (Contrastive Language-Assay Molecule Pre-Training): natural language to predict the most relevant molecule, given a textual description of a bioassay, without training samples. [[Paper](https://arxiv.org/abs/2303.03363v1)].  
* [molplotly](https://github.com/wjm41/molplotly) - An add-on to Plotly built on RDKit which allows 2D images of molecules to be shown in Plotly figures when hovering over the data points.  
* [MolForge](https://github.com/knu-lcbc/MolForge) - Neural-machine-translation based models that translate a set of various structural fingerprints to conventional text-based molecular representations, such as SMILES and SELFIES.  [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00693-0)]  
* [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules) - Equivariant Diffusion for Molecule Generation in 3D. [[Paper](https://arxiv.org/abs/2203.17003)]  
* [SELFormer](https://github.com/hubiodatalab/selformer) - Molecular Representation Learning via SELFIES Language Models. [[Paper](https://arxiv.org/abs/2304.04662)]  
* [Regression Transformer](https://github.com/GT4SD/gt4sd-core/tree/main/examples/regression_transformer) - Concurrent sequence regression and generation for molecular language modelling. [[Paper](https://www.nature.com/articles/s42256-023-00639-z)]  
* [Bio-Diffusion](https://github.com/BioinfoMachineLearning/bio-diffusion) - A PyTorch hub of denoising diffusion probabilistic models designed to generate novel biological data. [[Paper](https://arxiv.org/abs/2302.04313)]
* [InstructMol](https://github.com/idea-xl/instructmol) - Multi-Modal integration for building a versatile and reliable molecular assistant in Drug Discovery. [[Paper](https://arxiv.org/abs/2311.16208)]  
# Proteins
* [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk) - A online database which provides open access to 992,316 protein structure predictions for the human proteome and other key proteins of interest, to accelerate scientific research.
* [AlphaFold](https://github.com/deepmind/alphafold) - Open source code for DeepMind's AlphaFold.
* [OpenFold](https://github.com/aqlaboratory/openfold) - Trainable, memory-efficient, and GPU-friendly PyTorch reproduction of AlphaFold 2.  
* [AlphaFold - single sequence input](https://colab.research.google.com/github/sokrypton/af_backprop/blob/beta/examples/AlphaFold_single.ipynb) - A Colab notebook to predict the protein structure from a single sequence (for educational purposes only). 
* [ColabFold](https://github.com/sokrypton/ColabFold) - Making Protein folding accessible to all via Google Colab. [[Article](https://www.nature.com/articles/s41592-022-01488-1)]
* [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) - Running ColabFold on your local PC.  
* [Meaningful Protein Representation](https://github.com/MachineLearningLifeScience/meaningful-protein-representations) - Learning meaningful representations of protein sequences using a VAE. [[Article](https://www.nature.com/articles/s41467-022-29443-w)] [[Paper](https://arxiv.org/pdf/2012.02679.pdf)]. 
* [TAPE](https://github.com/songlab-cal/tape) - Tasks Assessing Protein Embeddings (TAPE), a set of five biologically relevant semi-supervised learning tasks spread across different domains of protein biology. [[Paper](https://arxiv.org/pdf/1906.08230.pdf)]
* [FastFold](https://github.com/hpcaitech/FastFold) - Optimizing Protein Structure Prediction Model Training and Inference on GPU Clusters.  
* [ESM](https://github.com/facebookresearch/esm) - Pretrained language models that enable zero-shot prediction of the effects of mutations on protein function. [[Paper](http://proceedings.neurips.cc/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)]  
    * [ESMFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb#scrollTo=POQBeXf2Xoxo) - Colab notebook for the "Language models of protein sequences at the scale of evolution enable accurate structure prediction" [paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1).  
* [Protein Sequence Embedding](https://github.com/tbepler/protein-sequence-embedding-iclr2019) - Learning protein sequence embeddings using information from structure. [[Paper](https://arxiv.org/pdf/1902.08661v2.pdf)]  
* [IdpGAN](https://github.com/feiglab/idpgan) - A GAN to generate different 3D conformations for intrinsically disordered proteins given their sequences. [[Article](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1.full)]  
* [PocketMiner](https://github.com/Mickdub/gvp/tree/pocket_pred) - A tool for predicting the locations of cryptic pockets from single protein structures. [[Paper](https://www.biorxiv.org/content/10.1101/2022.06.28.497399v2)]  
* [progen2](https://github.com/salesforce/progen) - A suite of open-sourced projects and models for protein engineering and design. [[Paper](https://arxiv.org/pdf/2206.13517.pdf)]  
* [TransformerCPI](https://github.com/lifanchen-simm/transformerCPI) - Improving compound–protein interaction prediction by sequence-based Deep Learning with self-attention mechanism and label reversal experiments. [[Paper](https://academic.oup.com/bioinformatics/article/36/16/4406/5840724)]  
* [Graphein](https://github.com/a-r-j/graphein) - A Python package which provides functionality for producing a number of types of graph-based representations of proteins, compatible with standard geometric Deep Learning library formats, as well as graph objects designed for ease of use with popular Deep Learning libraries.  
* [EvoBind](https://github.com/patrickbryant1/EvoBind) - In silico directed evolution of peptide binders with AlphaFold2. [[Paper](https://www.biorxiv.org/content/10.1101/2022.07.23.501214v1)]  
* [alphafold_finetune](https://github.com/phbradley/alphafold_finetune) - Python code for fine-tuning AlphaFold to perform protein-peptide binding predictions.  
* [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) - A deep unsupervised language model for protein design. [[Article](https://www.nature.com/articles/s41467-022-32007-7)]  
* [Bio Embeddings](https://github.com/sacdallago/bio_embeddings) - General purpose Python embedders based on open models trained on biological sequence representations. [[Paper](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.113)]  
* [Uni-Fold](https://github.com/dptech-corp/Uni-Fold) - An Open Source platform for developing protein models beyond AlphaFold. [[Paper](https://www.biorxiv.org/content/10.1101/2022.08.04.502811v1)]  
* [AF2Rank](https://github.com/jproney/AF2Rank) - State-of-the-art estimation of protein model accuracy using AlphaFold. [[Paper](https://www.biorxiv.org/content/10.1101/2022.03.11.484043v3)]  
* [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Robust Deep Learning based protein sequence design. [[Paper]](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)
  * [LigandMPNN](https://github.com/dauparas/LigandMPNN) - Atomic context-conditioned protein sequence design. [[Paper](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1)]  
* [DPAM](https://github.com/CongLabCode/DPAM) - A Domain Parser for AlphaFold Models. [[Paper](https://www.biorxiv.org/content/10.1101/2022.09.22.509116v1.full)]  
* [ModelAngelo](https://github.com/3dem/model-angelo) - Automatic atomic model building program for Electron cryo-microscopy (cryo-EM) maps. [[Paper](https://arxiv.org/abs/2210.00006)]  
* [DiffDock](https://github.com/gcorso/DiffDock) - Diffusion steps, twists, and turns for Molecular Docking. [[Paper](https://arxiv.org/abs/2210.01776)].  
* [MoLPC](https://github.com/patrickbryant1/MoLPC) - Predicting the structure of large protein complexes using AlphaFold and Monte Carlo tree search. [[Article](https://www.nature.com/articles/s41467-022-33729-4)]  
* [foldingdiff](https://github.com/microsoft/foldingdiff) - A diffusion model for generating novel protein backbone structures. [[Paper](https://arxiv.org/abs/2209.15611)]  
* [ProGen](https://github.com/salesforce/progen) - Suite of open-sourced projects and models for protein engineering and design. [[Paper](https://arxiv.org/abs/2004.03497)]  
* [DeepAb](https://github.com/RosettaCommons/DeepAb) - Antibody structure prediction using interpretable Deep Learning. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666389921002804)]  
* [cdna-display-proteolysis-pipeline](https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline) - Mega-scale experimental analysis of protein folding stability in biology and protein design.  [[Paper](https://www.biorxiv.org/content/10.1101/2022.12.06.519132v3.full)]  
* [PDBench](https://github.com/wells-wood-research/PDBench) - A dataset and software package for evaluating fixed-backbone sequence design algorithms. [[Paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad027/6986968?login=false)]  
* [vcMSA](https://github.com/clairemcwhite/vcmsa) - A Python library to run **v**ector **c**lustering **M**ultiple **S**equence **A**lignment. [[Paper](https://www.biorxiv.org/content/10.1101/2022.10.21.513099v1.full)]  
* [Ankh](https://github.com/agemagician/Ankh) - A optimized Protein Language Model. [[Paper](https://arxiv.org/abs/2301.06568v1)]  
* [TorchProtein](https://torchprotein.ai/) - A Machine Learning library for protein science, built on top of TorchDrug.  
* [GearNet](https://github.com/DeepGraphLearning/GearNet) - Geometric pretraining methods for Protein Structure Representation Learning. [[Paper](https://arxiv.org/abs/2203.06125)]  
* [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) - Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models. [[Paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)]  
* [TargetGAN](https://github.com/viko-3/TargetGAN) - A deep generative model for drug design from protein target sequence. [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00702-2)]  
* [Iterative_masking](https://github.com/Bitbol-Lab/Iterative_masking) - An iterative method that directly employs the masked language modeling objective to generate sequences using a [MSA Transformer](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v3.full). [[Paper](https://elifesciences.org/articles/79854)]
* [protpardelle](https://github.com/ProteinDesignLab/protpardelle) - An all-atom protein generative model. [[Paper](https://www.biorxiv.org/content/10.1101/2023.05.24.542194v1.full)]
* [ProteinGLUE](https://github.com/ibivu/protein-glue) - A multi-task benchmark suite for self-supervised protein modeling. [[Paper](https://www.nature.com/articles/s41598-022-19608-4)]
* [MassiveFold](https://github.com/GBLille/MassiveFold) - A tool that allows to massively expand the sampling of structure predictions by improving the computing of AlphaFold based predictions. [[Paper](https://www.nature.com/articles/s43588-024-00714-4)]  
# Cheminformatics
* [DRFP](https://github.com/reymond-group/drfp) - An NLP-inspired chemical reaction fingerprint based on basic set arithmetic. [[Article](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c)]. 
* [DeepChem](https://github.com/deepchem/deepchem) - A high quality Open Source toolchain that democratizes the use of Deep Learning in drug discovery, materials science, quantum chemistry, and biology.
* [CompAugCycleGAN](https://github.com/qai222/CompAugCycleGAN) - Augmented CycleGAN used for generating chemical compositions. [[Article](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d1dd00044f)]  
* [Chemformer](https://github.com/MolecularAI/Chemformer) - A pre-trained transformer for computational chemistry.  
* [RDKit](https://www.rdkit.org/docs/index.html) - Open Source toolkit for cheminformatics and Machine Learning.  
* [Streamlit-app](https://github.com/suneelbvs/Streamlit-app) - A Streamlit web app for cheminformatics which includes also a RDKit cheatsheet.  
* [datamol](https://github.com/datamol-org/datamol) - A lightweight Python library to work with molecules, built on top of RDKit.   
* [rxn_yields](https://github.com/rxn4chemistry/rxn_yields/tree/master/) - Prediction of chemical reaction yields using Deep Learning and data augmentation strategies. [[Article](https://iopscience.iop.org/article/10.1088/2632-2153/abc81d)]  
* [gptchem](https://github.com/kjappelbaum/gptchem) - Using GPT-3 to solve Chemistry problems. [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/63eb5a669da0bc6b33e97a35)]  
* [protein_scoring](https://github.com/seanrjohnson/protein_scoring) - Computational Scoring and experimental evaluation of enzymes generated by Neural Networks. [[Paper](https://www.biorxiv.org/content/10.1101/2023.03.04.531015v1)]  
* [Jazzy](https://github.com/AstraZeneca/jazzy) - A Python library that allows to calculate a set of atomic/molecular descriptors which include the Gibbs free energy of hydration (kJ/mol), its polar/apolar components, and the hydrogen-bond strength of donor and acceptor atoms using either SMILES or MOL/SDF inputs.
* [CRISPRi](https://github.com/BarquistLab/CRISPRi_guide_efficiency_bacteria) - Improved prediction of bacterial CRISPRi guide efficiency from depletion screens through mixed-effect Machine Learning and data integration. [[Paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03153-y)]  
# Drug Discovery
* [TorchDrug](https://github.com/DeepGraphLearning/torchdrug) - A powerful and flexible PyTorch-based Deep Learning platform for drug discovery.  
* [COVID-19 Multi-Targeted Drug Repurposing Using Few-Shot](https://github.com/XieResearchGroup/MolGNN_fewshot) - PyTorch implementation of MolGNN Few-shot. [[Article](https://www.frontiersin.org/articles/10.3389/fbinf.2021.693177/full)] [[Paper](https://www.researchgate.net/publication/352403842_COVID-19_Multi-Targeted_Drug_Repurposing_Using_Few-Shot_Learning)]  
* [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix) - A Bio-Computing Platform featuring Large-Scale Representation Learning and Multi-Task Deep Learning.  
* [liGAN](https://github.com/mattragoza/liGAN) - Deep generative models of 3D grids for structure-based drug discovery. [[Article](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a)] [[Paper](https://arxiv.org/pdf/2110.15200v2.pdf)]   
* [LIMO](https://github.com/Rose-STL-Lab/LIMO) - **L**atent **I**nceptionism for targeted **MO**lecule Generation: a generative model for drug discovery. [[Paper](https://arxiv.org/pdf/2206.09010.pdf)]  
* [DelFTa](https://github.com/josejimenezluna/delfta) - Δ-Quantum Machine Learning for medicinal chemistry. [[Paper](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp00834c)]  
* [Fréchet ChemNet Distance](https://github.com/bioinf-jku/FCD) - Fréchet ChemNet Distance: a quality measure for generative models for molecules. [[Paper](https://arxiv.org/pdf/1803.09518v3.pdf)].   
* [DrugOOD](https://github.com/tencent-ailab/DrugOOD) - A systematic OOD (Out-Of-Distribution) dataset curator and benchmark for AI-aided drug discovery. [[Paper](https://arxiv.org/abs/2201.09637)]  
* [PIGNet](https://github.com/ACE-KAIST/PIGNet) - a Physics Informed Deep Learning model toward generalized drug-target interaction predictions. [[Paper](https://pubs.rsc.org/en/content/articlelanding/2022/SC/D1SC06946B)]  
* [REINVENT](https://github.com/MolecularAI/Reinvent/tree/reinvent.v.2.0) - An AI tool for de novo drug design. [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/60c74fb60f50db7e51397450)]  
* [MolScore](https://github.com/MorganCThomas/MolScore) - An automated scoring function to facilitate and standardize evaluation of goal-directed generative models for de novo molecular design. [[Article](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00516-0)]  
* [SMILES-RNN](https://github.com/MorganCThomas/SMILES-RNN) - A SMILES-based recurrent neural network used for de novo molecule generation with several reinforcement learning algorithms available for molecule optimization. [[Article](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00646-z)] 
* [DiffLinker](https://github.com/igashov/difflinker) - Equivariant 3D-Conditional Diffusion Model for molecular linker design. [[Paper](https://arxiv.org/abs/2210.05274)]  
* [SQUID](https://github.com/keiradams/squid) - Equivariant shape-conditioned generation of 3D molecules for Ligand-Based Drug Design. [[Paper](https://arxiv.org/abs/2210.04893)]  
* [DiffSBDD](https://github.com/arneschneuing/diffsbdd) - A Euclidean diffusion model for structure-based drug design. [[Paper](https://arxiv.org/abs/2210.13695)]  
* [MF-PCBA](https://github.com/davidbuterez/mf-pcba) - Multi-fidelity high-throughput screening benchmarks for drug discovery and Machine Learning. [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/636fa49b80c9bfb4dc944c1c)]  
* [Deep Surrogate Docking](https://github.com/ryienh/graph-dock) - Accelerating automated Drug Discovery with Graph Neural Networks. [[Paper](https://arxiv.org/abs/2211.02720)]  
* [HGAN-DTI](https://github.com/Zora-LM/HGAN-DTI) - Heterogeneous Graph Attention Network for Drug-Target Interaction Prediction. [[Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557346)]  
* [MolSkill](https://github.com/microsoft/molskill) - Learning chemical intuition from humans in the loop. [[Paper](https://chemrxiv.org/engage/chemrxiv/article-details/63f89282897b18336f0c5a55)]  
* [AI-Bind](https://github.com/Barabasi-Lab/AI-Bind/) - Interpretable AI pipeline improving binding predictions for novel protein targets and ligands. [[Paper](https://www.nature.com/articles/s41467-023-37572-z)]  
# Datasets
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
# Explainable AI
* [Interpretable and Explainable Machine Learning for Materials Science and Chemistry](https://pubs.acs.org/doi/10.1021/accountsmr.1c00244) - Interpretable and Explainable Machine Learning applied to materials science and chemistry.
* [exmol](https://github.com/ur-whitelab/exmol) - Explainer for black box models that predict molecule properties. [[Article](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d1sc05259d)]  
* [BERTology](https://github.com/salesforce/provis) - Interpreting Attention in Protein Language Models. [[Paper](https://arxiv.org/pdf/2006.15222v3.pdf)]  
* [DRPreter](https://github.com/babaling/DRPreter) - Interpretable anticancer drug response prediction using Knowledge-Guided Graph Neural Networks and Transformer. [[Paper](https://www.mdpi.com/1422-0067/23/22/13919)]  
# Other
* [nglview](https://github.com/nglviewer/nglview) - A Jupyter widget to interactively view molecular structures and trajectories.  
* [Panel-Chemistry](https://github.com/marcskovmadsen/panel-chemistry) - Easy exploratory data analysis and build powerful data and viz tools within the domain of Chemistry using Python and HoloViz Panel.  
* [libmolgrid](https://github.com/gnina/libmolgrid) - A comprehensive library for fast, GPU accelerated molecular gridding for Deep Learning workflows. [[Paper](https://arxiv.org/pdf/1912.04822.pdf)]  
* [stmol](https://github.com/napoles-uach/stmol) - A component for building interactive molecular 3D visualizations within Streamlit web applications.  
* [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) - An add-on and set of pre-made nodes for Blender & Blender’s Geometry Nodes, to import, animate and manipulate molecular data.  
* [Jupyter Dock](https://github.com/AngelRuizMoreno/Jupyter_Dock) - Molecular Docking integrated in Jupyter notebooks.  
* [AugLiChem](https://github.com/BaratiLab/AugLiChem) - A data augmentation library for chemical structures. [[Paper](https://arxiv.org/abs/2111.15112)]  
* [Chemiscope](https://github.com/lab-cosmo/chemiscope) - An interactive structure/property explorer for materials and molecules. While the core project is implemented in a different programming language, it has been added to this list because it provides Python extensions that allow using it within a Jupyter or Colab notebook.  
* [ReMODE](https://github.com/micahwang/ReMode/tree/master) - A Deep Learning-based web server for target-specific drug design. [[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00665-w)]  
