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

def total_hydrophobicity(sequence, type_of_h):
    total_phob = 0
    for letter in sequence:
        if (type_of_h == "positive"):
            if (letter in SIDE_CHAIN_CHARGE and SIDE_CHAIN_CHARGE[letter] > 0):
                total_phob += SIDE_CHAIN_CHARGE[letter]

        if (type_of_h == "negative"):
            if (letter in SIDE_CHAIN_CHARGE and SIDE_CHAIN_CHARGE[letter] < 0):
                total_phob += SIDE_CHAIN_CHARGE[letter]

    return total_phob


def average_side_chain_charge(sequence):
    ave_charge = 0
    non_counted = 0
    for letter in sequence:
        if (letter in SIDE_CHAIN_CHARGE):
            ave_charge+= SIDE_CHAIN_CHARGE[letter]
        else:
            non_counted += 1.0

    return ave_charge / (len(sequence) - non_counted)


def total_charge(sequence,type_of_charge):
    total_charge = 0
    for letter in sequence:
        if(type_of_charge == "positive"):
            if (letter in SIDE_CHAIN_CHARGE and SIDE_CHAIN_CHARGE[letter]>0):
                total_charge += SIDE_CHAIN_CHARGE[letter]

        if(type_of_charge == "negative"):
            if (letter in SIDE_CHAIN_CHARGE and SIDE_CHAIN_CHARGE[letter]<0):
                total_charge += SIDE_CHAIN_CHARGE[letter]

    return total_charge


def global_feature_dict(record, amount_start = 50, amount_in_end=50, global_bipeptide = True, local_bipeptide=False, aromaticity = True,instability = True, average_h =  True, side_charge_ave = True, gravy = False):
    result = defaultdict(float)
    sequence = str(record.seq)
    simplified_sequence = re.sub('[XU]', '', sequence)
    simplified_sequence = re.sub('[B]', 'D', simplified_sequence)

    #Global Features
    result["sequence_length"] = len(sequence)
    result = feature_extractor(result,sequence, simplified_sequence,"global", bipeptide = global_bipeptide, aromaticity=aromaticity, instability = instability,average_h=average_h,side_charge_ave=side_charge_ave, gravy =gravy )

    #Start Features
    start_pattern = sequence[:amount_start]
    simplified_start_pattern = simplified_sequence[:amount_start]
    result = feature_extractor(result, start_pattern, simplified_start_pattern, "start",bipeptide = local_bipeptide, secondary_struct=False, aromaticity=aromaticity, instability = instability,average_h=average_h,side_charge_ave=side_charge_ave, gravy =gravy)

    #End Features
    end_pattern = sequence[-amount_in_end:]
    simplified_end_pattern = simplified_sequence[-amount_in_end:]
    result = feature_extractor(result, end_pattern, simplified_end_pattern, "end",  bipeptide = local_bipeptide, secondary_struct=False, aromaticity=aromaticity, instability = instability,average_h=average_h,side_charge_ave=side_charge_ave, gravy =gravy)

    return result




def feature_extractor(result, sequence,simplified_sequence, suffix, bipeptide=False, secondary_struct=True, side_charge = True,peptide=True, aromaticity = True, instability = True, average_h=True,side_charge_ave=True, gravy = False):

    if(peptide):
        for acid in AMINO_ACIDS:
            result["{}_composition_{}".format(acid,suffix)] = aminoacid_composition(sequence, acid)

    if(bipeptide):
        for acid1 in AMINO_ACIDS:
            for acid2 in AMINO_ACIDS:
                result["{}{}_composition_{}".format(acid1,acid2,suffix)]=aminoacid_composition(sequence,acid1+acid2)

    if(average_h):
        result["hydrophobicity_{}".format(suffix)] = average_hydrophobicity(sequence)

    result["total_positive_hydrophobicity_{}".format(suffix)] = total_hydrophobicity(sequence,"positive")
    result["total_negative_hydrophobicity_{}".format(suffix)] = total_hydrophobicity(sequence,"negative")

    if(side_charge):
        if(side_charge_ave):
            result["side_chain_charge_{}".format(suffix)] = average_side_chain_charge(sequence)
        result["total_positive_charge_{}".format(suffix)] = total_charge(sequence,"positive")
        result["total_negative_charge_{}".format(suffix)] = total_charge(sequence, "negative")

    tools = ProteinAnalysis(sequence)
    tools_simplified_sequence = ProteinAnalysis(simplified_sequence)

    if(secondary_struct):
        helix, turn, sheet = tools.secondary_structure_fraction()
        result["helix_{}".format(suffix)] = helix
        result["turn_{}".format(suffix)] = turn
        result["sheet_{}".format(suffix)] = sheet

    if(aromaticity):
        result["aromaticity_{}".format(suffix)] = tools.aromaticity()
    result["isoelectric_point_{}".format(suffix)] = tools.isoelectric_point()

    result["molecular_weight_{}".format(suffix)] = tools_simplified_sequence.molecular_weight()

    if(instability):
        result["instability_index_{}".format(suffix)] = tools_simplified_sequence.instability_index()

    if(gravy):
        result["gravy_{}".format(suffix)] = tools_simplified_sequence.gravy()

    return result





















def global_and_sliding_window_feature_dict(record= None, window = 50, bipeptide = False, aromaticity = True,instability = True ):
    if(record==None):
        raise AttributeError("record needs to be provided")

    result = defaultdict(float)
    sequence = str(record.seq)
    simplified_sequence = re.sub('[XU]', '', sequence)
    simplified_sequence = re.sub('[B]', 'D', simplified_sequence)

    seq_length =len(sequence)
    result["sequence_length"] = seq_length

    result = feature_extractor(result,sequence, simplified_sequence,"global",bipeptide, instability = instability, aromaticity = aromaticity)

    # Start Features
    start_pattern = sequence[:50]
    simplified_start_pattern = simplified_sequence[:50]
    result = feature_extractor(result, start_pattern, simplified_start_pattern, "start", bipeptide=False, secondary_struct=False, instability = instability, aromaticity = aromaticity)

    # End Features
    end_pattern = sequence[-50:]
    simplified_end_pattern = simplified_sequence[:50]
    result = feature_extractor(result, end_pattern, simplified_end_pattern, "end", bipeptide=False,secondary_struct=False, instability = instability, aromaticity = aromaticity)


    if(seq_length > window):
        count = 50

        while(count+window < seq_length-50):
            segment = sequence[count:count+window]
            simplified_segment = re.sub('[XU]', '', segment)
            simplified_segment= re.sub('[B]', 'D', simplified_segment)
            result = feature_extractor(result,segment,simplified_segment,count,bipeptide=False,secondary_struct=False,side_charge = False,peptide=False)
            count += window

        segment = sequence[count:]
        simplified_segment = re.sub('[XU]', '', segment)
        simplified_segment = re.sub('[B]', 'D', simplified_segment)
        result = feature_extractor(result, segment, simplified_segment, count,bipeptide=False,secondary_struct=False,side_charge = False,peptide=False)

    return result





