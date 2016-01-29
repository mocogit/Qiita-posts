#
#   gen_cmajor.py   
#       date. 1/22/2016, 1/23, 1/25
#

from __future__ import print_function
import numpy as np

scale_names = ['Cmj', 'Gmj', 'Dmj', 'Amj', 'Emj', 'Amn', 'Emn', 'Bmn', 'Fsmn', 'Csmn']

cmj_set = [3, 5, 7, 8, 10, 12, 14]
cmj_base = [3, 15]
amn_base = [12, 24]

gmj_set = [3, 5, 7, 9, 10, 12, 14]
gmj_base = [10, 22]
emn_base = [7, 19]

dmj_set = [4, 5, 7, 9, 10, 12, 14]
dmj_base = [5, 17]
bmn_base = [14, 26]

amj_set = [4, 5, 7, 9, 11, 12, 14] 
amj_base = [12, 24]
fsmn_base = [9, 21]

emj_set = [4, 6, 7, 9, 11, 12, 14] 
emj_base = [7, 19]
csmn_base = [4, 16] 

scale_db = [
    [cmj_set, cmj_base, amn_base],
    [gmj_set, gmj_base, emn_base],
    [dmj_set, dmj_base, bmn_base],
    [amj_set, amj_base, fsmn_base],
    [emj_set, emj_base, csmn_base]
    ]


def prep_2x_scale(scale):
    scale2 = []
    for i in range(len(scale)):
        scale2.append(scale[i])
    for i in range(len(scale)):
        scale2.append((scale[i] + 12))
        
    return scale2

def gen_seq(m_len, base, scale_list):
    seq = []
    direct = np.random.randint(7, size=m_len)
    for i in range(m_len):
        if i == 0:
            cur_key = base[1]
            seq.append(cur_key)
            
        else:
            cur_keyi = scale_list.index(cur_key)
            tdirect = direct[i]
            if tdirect < 3:   # scale descending
                next_keyi = cur_keyi - 1      
            elif tdirect < 4: # keep current note
                next_keyi = cur_keyi  
            else:             # scale ascending
                next_keyi = cur_keyi + 1
            
            if ((next_keyi >= 0) and (next_keyi < len(scale_list))):
                cur_key = scale_list[next_keyi]
            
            seq.append(cur_key)
    
    return seq

def prep_keyset(key_str):
    idx = scale_names.index(key_str)
    myset = scale_db[(idx % 5)][0]
    is_minor = idx // 5
    mybase = scale_db[(idx % 5)][is_minor+1]
    myscale2 = prep_2x_scale(myset)
    
    return myscale2, mybase
    
def check_proper_ending(arg_scale, base):
    last_key = arg_scale[-1]
    if last_key in base:
        proper = True
    else:
        proper = False
    
    return proper
    
            
if __name__ == '__main__':
    np.random.seed(seed=20160122)
    m_len = 20
    num = 2000
    ending_check_sw = True
    
    if ending_check_sw == True:
        mycount = 0
        while True:
            key_index = np.random.randint(10)
            key_str = scale_names[key_index]
            
            myset2, mybase = prep_keyset(key_str)
            myscale = gen_seq(m_len, mybase, myset2)
            
            if check_proper_ending(myscale, mybase) == True:
                print(str(myscale).strip('[]')+',', key_str)
                mycount += 1
            if mycount == num:
                break
    
    else:
    
        for i in range(num):
            key_index = np.random.randint(10)
            key_str = scale_names[key_index]
        
            myset2, mybase = prep_keyset(key_str)
            myscale = gen_seq(m_len, mybase, myset2)
        
       
            print(str(myscale).strip('[]')+',', key_str)
