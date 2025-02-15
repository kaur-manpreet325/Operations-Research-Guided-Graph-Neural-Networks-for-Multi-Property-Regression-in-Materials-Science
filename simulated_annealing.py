import json
import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from tensorflow.keras.optimizers import Adam
from pymatgen.core import Structure
from scipy.stats import pearsonr
import pickle

# Set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_data = load_data('train.json')
valid_data = load_data('valid.json')
test_data = load_data('test.json')

# Prepare data
def prepare_data(data):
    structures = []
    targets = []
    for item in data:
        try:
            structure = Structure.from_dict(item['structure'])
        except:
            structure = Structure(item['structure']['lattice']['matrix'],
                                  item['structure']['species'],
                                  item['structure']['coords'])
        structures.append(structure)
        targets.append([
            item.get('band_gap', 0),
            item.get('formation_energy_per_atom', 0),
            item.get('energy_above_hull', 0)
        ])
    return structures, np.array(targets)

train_structures, train_targets = prepare_data(train_data)
valid_structures, valid_targets = prepare_data(valid_data)
test_structures, test_targets = prepare_data(test_data)

# Normalize targets
scaler = StandardScaler()
train_targets_normalized = scaler.fit_transform(train_targets)
valid_targets_normalized = scaler.transform(valid_targets)
test_targets_normalized = scaler.transform(test_targets)

# Initialize MEGNet model
def initialize_model(lr, npass, nblocks):
    nfeat_bond = 20
    r_cutoff = 5
    gaussian_centers = np.linspace(0, r_cutoff, nfeat_bond)
    gaussian_width = 0.5
    graph_converter = CrystalGraph(bond_converter=GaussianDistance(gaussian_centers, gaussian_width))

    model = MEGNetModel(
        nfeat_bond,
        2,  # nfeat_global
        nblocks=nblocks,
        lr=lr,
        n1=64,
        n2=32,
        n3=16,
        npass=npass,
        ntarget=3,
        graph_converter=graph_converter
    )
    model.model.compile(optimizer=Adam(lr), loss='mse')
    return model

# Evaluation function
def evaluate_model(model, structures, targets):
    pred = model.predict_structures(structures)
    pred_denormalized = scaler.inverse_transform(pred)

    # Remove NaN values
    valid_indices = ~np.isnan(pred_denormalized).any(axis=1) & ~np.isnan(targets).any(axis=1)
    pred_denormalized = pred_denormalized[valid_indices]
    targets = targets[valid_indices]

    if len(pred_denormalized) == 0:
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), [np.nan, np.nan, np.nan]

    mae = mean_absolute_error(targets, pred_denormalized, multioutput='raw_values')
    r2 = r2_score(targets, pred_denormalized, multioutput='raw_values')
    pearson = []
    for i in range(3):
        if np.all(targets[:, i] == targets[0, i]) or np.all(pred_denormalized[:, i] == pred_denormalized[0, i]):
            pearson.append(np.nan)
        else:
            pearson.append(pearsonr(targets[:, i], pred_denormalized[:, i])[0])
    return mae, r2, pearson

# Logging function
def log_results(message, file):
    print(message)
    file.write(message + "\n")

# Checkpoint saving function
def save_checkpoint(iteration, evaluation, params, mae):
    checkpoint = {
        'iteration': iteration,
        'evaluation': evaluation,
        'params': params,
        'mae': mae
    }
    with open('checkpoint_SA.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

# Checkpoint loading function
def load_checkpoint():
    try:
        with open('checkpoint_SA.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Simulated Annealing
def simulated_annealing(num_iterations=5, evaluations_per_iteration=5, initial_temp=1.0, cooling_rate=0.95):
    best_mae = float('inf')
    best_params = None
    current_temp = initial_temp

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        start_iteration = checkpoint['iteration']
        start_evaluation = checkpoint['evaluation']
        best_params = checkpoint['params']
        best_mae = checkpoint['mae']
    else:
        start_iteration = 0
        start_evaluation = 0

        # Initial random parameters
        best_params = {
            'epochs': random.randint(50, 250),
            'lr': random.choice([1e-5, 1e-2, 5e-5, 1e-4, 1e-3]),
            'batch_size': random.choice([16, 32, 64, 128]),
            'npass': random.randint(1, 5),
            'nblocks': random.randint(1, 5)
        }

    with open('optimization_results_SA.txt', 'a') as f:
        for i in range(start_iteration, num_iterations):
            for j in range(start_evaluation, evaluations_per_iteration):
                log_results(f"Iteration {i + 1} Evaluation {j + 1}:", f)

                # Propose a new set of parameters (neighbor solution)
                new_params = {
                    'epochs': best_params['epochs'] + random.randint(-20, 20),
                    'lr': random.choice([1e-5, 1e-2, 5e-5, 1e-4, 1e-3]),  # keep lr random
                    'batch_size': random.choice([16, 32, 64, 128]),  # keep batchsize random
                    'npass': best_params['npass'] + random.randint(-1, 1),
                    'nblocks': best_params['nblocks'] + random.randint(-1, 1)
                }

                # Ensure parameters stay within reasonable bounds
                new_params['epochs'] = max(50, min(new_params['epochs'], 250))
                new_params['npass'] = max(1, min(new_params['npass'], 5))
                new_params['nblocks'] = max(1, min(new_params['nblocks'], 5))

                log_results(f"Hyperparameters:", f)
                log_results(f"Epochs: {new_params['epochs']}", f)
                log_results(f"Learning rate: {new_params['lr']:.6f}", f)
                log_results(f"Batch size: {new_params['batch_size']}", f)
                log_results(f"Passes: {new_params['npass']}", f)
                log_results(f"Blocks: {new_params['nblocks']}\n", f)

                model = initialize_model(new_params['lr'], new_params['npass'], new_params['nblocks'])

                model.train(
                    train_structures,
                    train_targets_normalized,
                    validation_structures=valid_structures,
                    validation_targets=valid_targets_normalized,
                    epochs=new_params['epochs'],
                    batch_size=new_params['batch_size']
                )

                mae, r2, pearson = evaluate_model(model, valid_structures, valid_targets)
                overall_mae = np.nanmean(mae)
                overall_r2 = np.nanmean(r2)
                overall_pearson = np.nanmean(pearson)

                if np.isnan(overall_mae):
                    log_results("Warning: NaN values encountered. Skipping this evaluation.", f)
                    continue

                log_results(f"Average Validation MAE: {overall_mae:.6f}", f)
                log_results(f"Average Validation R2: {overall_r2:.6f}", f)
                log_results(f"Average Validation Pearson: {overall_pearson:.6f}\n", f)

                # Acceptance criterion
                if overall_mae < best_mae:
                    # Accept the new solution
                    best_mae = overall_mae
                    best_params = new_params
                    log_results("New best model found!\n", f)
                else:
                    # Calculate the probability of accepting the worse solution
                    delta_mae = overall_mae - best_mae
                    acceptance_probability = np.exp(-delta_mae / current_temp)

                    # Accept the worse solution with a certain probability
                    if random.random() < acceptance_probability:
                        best_params = new_params  # still update even if worse.
                        log_results("Worse model accepted based on simulated annealing criteria.\n", f)

                # Save checkpoint
                save_checkpoint(i, j + 1, best_params, best_mae)

            # Reset start_evaluation for the next iteration
            start_evaluation = 0

            # Cool down the temperature
            current_temp *= cooling_rate

    return best_params

# Main execution
start_time = time.time()
best_params = simulated_annealing()
end_time = time.time()

# Final model training and evaluation
epochs, lr, batch_size, npass, nblocks = best_params['epochs'], best_params['lr'], best_params['batch_size'], best_params['npass'], best_params['nblocks']
final_model = initialize_model(lr, npass, nblocks)

final_model.train(
    train_structures,
    train_targets_normalized,
    validation_structures=valid_structures,
    validation_targets=valid_targets_normalized,
    epochs=epochs,
    batch_size=batch_size
)

test_mae, test_r2, test_pearson = evaluate_model(final_model, test_structures, test_targets)

overall_test_mae = np.nanmean(test_mae)
overall_test_r2 = np.nanmean(test_r2)
overall_test_pearson = np.nanmean(test_pearson)

with open('optimization_results_SA.txt', 'a') as f:
    log_results("\nFinal Model Evaluation on Test Set:", f)

    # Log the best hyperparameters
    log_results("\nBest Hyperparameters Used for Test Set Evaluation:", f)
    log_results(f"Epochs: {epochs}", f)
    log_results(f"Learning rate: {lr:.6f}", f)
    log_results(f"Batch size: {batch_size}", f)
    log_results(f"Passes: {npass}", f)
    log_results(f"Blocks: {nblocks}\n", f)

    for i, prop in enumerate(['band_gap', 'formation_energy_per_atom', 'energy_above_hull']):
        log_results(f"MAE for {prop}: {test_mae[i]:.6f}", f)
        log_results(f"R2 for {prop}: {test_r2[i]:.6f}", f)
        log_results(f"Pearson for {prop}: {test_pearson[i]:.6f}", f)
    log_results(f"Average Test MAE: {overall_test_mae:.6f}", f)
    log_results(f"Average Test R2: {overall_test_r2:.6f}", f)
    log_results(f"Average Test Pearson: {overall_test_pearson:.6f}", f)
    log_results(f"Total optimization time: {end_time - start_time:.1f} seconds", f)
