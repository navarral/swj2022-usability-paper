# Designing a usability study for health data researchers
[Special Issue on Semantic Web Meets Health Data Management](http://www.semantic-web-journal.net/blog/call-papers-special-issue-semantic-web-meets-health-data-management)

## Authors
Albert Navarro-Gallinad, Fabrizio Orlandi, and Declan O’Sullivan

ADAPT Centre for Digital Content, Trinity College Dublin, Dublin

Contact: albert.navarro@adaptcentre.ie

## Article's Data
The `data/` folder contains the necessary data to reproduce all the analysis, 
figures and tables from the usability testing experiments presented in the article.

The data files are organised as follows:

### Structure

```
.
├── data\
│   ├── PSSUQ\
│   │   ├── comments\
│   │   │   └── P{id}_Nov21_PSSUQ_Comments.odt
│   │   ├── scores\
│   │   │   └── Empty
│   │   ├── SERDIF_PSSUQ_P2.csv
│   │   └── README.md
│   ├── notes\
│   │   ├── P{id}_Notes.txt
│   │   └── README.md
│   ├── tasks\
│   │   ├── completion\
│   │   │   ├──  taskAssist.csv
│   │   │   ├──  taskAssistNavigationComplex.csv
│   │   │   ├──  taskAssistSystemIssue.csv
│   │   │   ├──  taskAssistTaskComplex.csv
│   │   │   ├──  taskSuccess.csv
│   │   │   ├──  taskSuccessAssistance.csv 
│   │   │   └──  README.md
│   │   ├── time\
│   │   │   └── TimeTaskP{id}.csv
│   │   └── README.md
│   ├── transcripts\
│   │   ├── raw\
│   │   │   ├──  P{id}_GMT{date}.odt
│   │   │   └──  README.md
│   │   ├── thematic-analysis\
│   │   │   ├── codebooks\
│   │   │   |   └── codebook_{date}_{iteration}.csv
│   │   │   ├── tags\
│   │   │   |   ├── csv\
│   │   │   |       └── all_tags_{date}_{iteration}.csv
│   │   │   |   ├── text\
│   │   │   |       └── all_tags_{date}_{iteration}.docx
│   │   │   |   └── README.md
│   │   │   ├── SERDIF_P2_ThematicAnalysis.sqlite3
│   │   │   └── README.md
│   │   └── README.md
│   └── README.md (The main readme)
└── README.md (The main readme)
```
## Code
### Figures

The figures in the article can be reproduced by running the `swj2022-figures.py` script.
The commands to run the Python file are the following:

1. [Download](https://github.com/navarral/swj2022-usability-paper/archive/refs/heads/main.zip) or clone the repository
2. Open a new terminal on the main project's folder
3. Run the following commands:
   1. `source venv/bin/activate`
   2. `pip install -r requirements.txt`
   3. `python swj2022-figures.py`
4. The generation of the article's figures can also be traced in the following [Jupyter notebook](https://github.com/navarral/swj2022-usability-paper/blob/main/swj2022-figures-notebook.ipynb).