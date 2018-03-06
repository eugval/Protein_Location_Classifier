from collections import defaultdict
from globals import AMINO_ACIDS, HYDROPHOBICITY, SIDE_CHAIN_CHARGE
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re


def aminoacid_composition(sequence, amino_acid,scope = "all",length =0):
   if(scope=="all"):
        return sequence.count(amino_acid)/len(sequence)
   elif(scope=="end"):
       if(len(sequence)<3):
           length = len(sequence)
       if(length <(len(sequence)-1)):
           length = int(len(sequence)/3.0)
       return sequence[-length:].count(amino_acid)/len(sequence[-length:])
   elif (scope == "start"):
       if(len(sequence)<3):
           length = len(sequence)
       if(length <(len(sequence)-1)):
           length = int(len(sequence)/3.0)
       return sequence[:length].count(amino_acid) / len(sequence[:length])


def average_hydrophobicity(sequence):
    ave_hydrophobicity = 0
    non_counted = 0
    for letter in sequence:
        if(letter in  HYDROPHOBICITY):
            ave_hydrophobicity+=HYDROPHOBICITY[letter]
        else:
            non_counted+=1.0

    return ave_hydrophobicity/(len(sequence)-non_counted)




def dictionary_feature_extractor(record):
    result = defaultdict(float)
    sequence = str(record.seq)
    simplified_sequence = re.sub('[XU]', '', sequence)
    simplified_sequence = re.sub('[B]', 'D', simplified_sequence)

    #Global features
    result["sequence_length"] = len(sequence)

    for acid in AMINO_ACIDS:
        result["global_{}_composition".format(acid)] = aminoacid_composition(sequence, acid)

    result["global_hydrophobicity"] = average_hydrophobicity(sequence)

    tools = ProteinAnalysis(sequence)
    tools_simplified_sequence= ProteinAnalysis(simplified_sequence)
    helix,turn,sheet = tools.secondary_structure_fraction()
    result["helix"] = helix
    result["turn"] =  turn
    result["sheet"] = sheet

    result["aromaticity"]=tools.aromaticity()
    result["isoelectric_point"] = tools.isoelectric_point()
    result["molecular_weight"] = tools_simplified_sequence.molecular_weight()
    result["instability_index"]=tools_simplified_sequence.instability_index()
    result["gravy"] = tools_simplified_sequence.gravy()
    

    #Start features
    start_pattern = sequence[:50]
    simplified_start_pattern = simplified_sequence[:50]


    for acid in AMINO_ACIDS:
        result["{}_composition_start".format(acid)] = aminoacid_composition(record.seq, acid, "start", 50)

    tools = ProteinAnalysis(start_pattern)
    tools_simplified_sequence = ProteinAnalysis(simplified_start_pattern)


    result["start_hydrophobicity"] = average_hydrophobicity(start_pattern)
    
    result["aromaticity_start"] = tools.aromaticity()
    result["isoelectric_point_start"] = tools.isoelectric_point()
    helix, turn, sheet = tools.secondary_structure_fraction()
    result["helix_start"] = helix
    result["turn_start"] = turn
    result["sheet_start"] = sheet

    result["molecular_weight_start"] = tools_simplified_sequence.molecular_weight()
    result["instability_index_start"] = tools_simplified_sequence.instability_index()
    result["gravy_start"] = tools_simplified_sequence.gravy()
    





    #End Features
    end_pattern = sequence[-50:]
    simplified_end_pattern = simplified_sequence[:50]

    tools = ProteinAnalysis(end_pattern)
    tools_simplified_sequence = ProteinAnalysis(simplified_end_pattern)

    for acid in AMINO_ACIDS:
        result["{}_composition_end".format(acid)] = aminoacid_composition(record.seq, acid, "end", 50)

    result["end_hydrophobicity"] = average_hydrophobicity(end_pattern)

    result["isoelectric_point_end"] = tools.isoelectric_point()
    result["aromaticity_end"] = tools.aromaticity()
    helix, turn, sheet = tools.secondary_structure_fraction()
    result["helix_end"] = helix
    result["turn_end"] = turn
    result["sheet_end"] = sheet
    
    result["molecular_weight_end"] = tools_simplified_sequence.molecular_weight()
    result["instability_index_end"] = tools_simplified_sequence.instability_index()
    result["gravy_end"] = tools_simplified_sequence.gravy()

    return result




def global_feature_dict(record, bipeptide = False):
    result = defaultdict(float)
    sequence = str(record.seq)
    simplified_sequence = re.sub('[XU]', '', sequence)
    simplified_sequence = re.sub('[B]', 'D', simplified_sequence)

    #Global Features
    result["sequence_length"] = len(sequence)
    result = feature_extractor(result,sequence, simplified_sequence,"global",bipeptide)

    #Start Features
    start_pattern = sequence[:50]
    simplified_start_pattern = simplified_sequence[:50]
    result = feature_extractor(result, start_pattern, simplified_start_pattern, "start",bipeptide)

    #End Features
    end_pattern = sequence[-50:]
    simplified_end_pattern = simplified_sequence[:50]
    result = feature_extractor(result, end_pattern, simplified_end_pattern, "end",bipeptide)

    return result



def global_and_sliding_window_feature_dict(record, window = 50, bipeptide = False):
    result = defaultdict(float)
    sequence = str(record.seq)
    simplified_sequence = re.sub('[XU]', '', sequence)
    simplified_sequence = re.sub('[B]', 'D', simplified_sequence)

    seq_length =len(sequence)
    result["sequence_length"] = seq_length

    result = feature_extractor(result,sequence, simplified_sequence,"global",bipeptide)

    if(seq_length > window):
        count = 0

        while(count+window < seq_length):
            segment = sequence[count:count+window]
            simplified_segment = re.sub('[XU]', '', segment)
            simplified_segment= re.sub('[B]', 'D', simplified_segment)
            result = feature_extractor(result,segment,simplified_segment,count,bipeptide)
            count += window

        segment = sequence[count:]
        simplified_segment = re.sub('[XU]', '', segment)
        simplified_segment = re.sub('[B]', 'D', simplified_segment)
        result = feature_extractor(result, segment, simplified_segment, count,bipeptide)

    return result


def feature_extractor(result, sequence,simplified_sequence, suffix, bipeptide):

    for acid in AMINO_ACIDS:
        result["{}_composition_{}".format(acid,suffix)] = aminoacid_composition(sequence, acid)

    if(bipeptide):
        for acid1 in AMINO_ACIDS:
            for acid2 in AMINO_ACIDS:
                result["{}{}_composition_{}".format(acid1,acid2,suffix)]=aminoacid_composition(sequence,acid1+acid2)

    result["hydrophobicity_{}".format(suffix)] = average_hydrophobicity(sequence)

    tools = ProteinAnalysis(sequence)
    tools_simplified_sequence = ProteinAnalysis(simplified_sequence)

    helix, turn, sheet = tools.secondary_structure_fraction()
    result["helix_{}".format(suffix)] = helix
    result["turn_{}".format(suffix)] = turn
    result["sheet_{}".format(suffix)] = sheet
    result["aromaticity_{}".format(suffix)] = tools.aromaticity()
    result["isoelectric_point_{}".format(suffix)] = tools.isoelectric_point()

    result["molecular_weight_{}".format(suffix)] = tools_simplified_sequence.molecular_weight()
    result["instability_index_{}".format(suffix)] = tools_simplified_sequence.instability_index()
    result["gravy_{}".format(suffix)] = tools_simplified_sequence.gravy()

    return result






