CLASSES = ["Cytosolic", "Mitochondrial", "Nucleic", "Secreted"]
CLASSES_S = ["Cyto", "Mito", "Nuc", "Secr"]
CLASS_INDICES ={"cyto":0,"mito":1,"nuc":2,"secr":3}

data_folder =  "Saved_Data/Features/"
models_folder = "Saved_Data/ModelOutputs/"
error_folder = "Saved_Data/ErrorAnalysis/"

LRCV ="LRCV"
RFCV = "RFCV"
GBCV = "GBCV"
SVMCV ="SVMCV"
ensembler = "Ensembler"
ensemblerCV = "ensembleClassifierCV"

model_types = [LRCV,RFCV,GBCV,SVMCV]

AMINO_ACIDS = ["A",
               "P",
               "B",
               "Q",
               "C",
               "R",
               "D",
               "S",
               "E",
               "T",
               "F",
               "U",
               "G",
               "V",
               "H",
               "W",
               "I",
               "Y",
               "K",
               "Z",
               "L",
               "X",
               "M",
               "N"]


SIDE_CHAIN_CHARGE = {"A":0,
               "P":0,
               "B":-0.5,
               "Q":0,
               "C":0,
               "R":1.0,
               "D":-1.0,
               "S":0,
               "E":-1.0,
               "T":0,
               "F":0,
               "U":0,
               "G":0,
               "V":0,
               "H":-0.9,
               "W":0,
               "I":0,
               "Y":0,
               "K":1.0,
               "Z":-0.5,
               "L":0,
               "X":0,
               "M":0,
               "N":0}

HYDROPHOBICITY = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"E":-3.5,"Q":-3.5,"G":-0.4,"H":-3.2,
                  "I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,
                  "W":-0.9,"Y":-1.3,"V":4.2}









