import os
import tensorflow as tf
import align_pdb_to_probes as align
from tqdm.keras import TqdmCallback
import numpy as np 
import json 

def predict():
    
    model = tf.keras.models.load_model("models/test_1.best.hdf5")
    print(model.summary())
    
    probe_file = "/y/people/jsd523/dev/nautilus_library_gen/probes/probe_C3C4O3_3.json"

    pdb_code = "1hr2"
    
    data = None
    with open(probe_file, "r", encoding="UTF-8") as json_file:
        data = json.load(json_file)
    
    pdb_gen = align.analyse_pdb(pdb_code, probe_data=data)
    if not pdb_gen: 
        return
    
    for generated_data in pdb_gen:
        probes, epsilon, phi, eta = generated_data

        sins = [np.sin(epsilon), np.sin(phi), np.sin(eta)]
        coss = [np.cos(epsilon), np.cos(phi), np.cos(eta)]

        probes = probes.reshape(1,-1)
        predicted_torisons = model.predict(probes)
        pred_sin = predicted_torisons[0][0]
        pred_cos = predicted_torisons[1][0]
           
        print(f"{'='*10}{pdb_code}{'='*10}")        
        print("SIN\tReal\tPred")
        print(f"EPS:\t {sins[0]:.2f}\t{pred_sin[0]:.2f}")
        print(f"PHI:\t {sins[1]:.2f}\t{pred_sin[1]:.2f}")
        print(f"ETA:\t {sins[2]:.2f}\t{pred_sin[2]:.2f}")

        print("COS\tReal\tPred")
        print(f"EPS:\t {coss[0]:.2f}\t{pred_cos[0]:.2f}")
        print(f"PHI:\t {coss[1]:.2f}\t{pred_cos[1]:.2f}")
        print(f"ETA:\t {coss[2]:.2f}\t{pred_cos[2]:.2f}")

        return
    

if __name__ == "__main__":
    predict()
