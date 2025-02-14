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
def save_checkpoint(generation, individual, params, mae):
    checkpoint = {
        'generation': generation,
        'individual': individual,
        'params': params,
        'mae': mae
    }
    with open('checkpoint_GA.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

# Checkpoint loading function
def load_checkpoint():
    try:
        with open('checkpoint_GA.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Genetic Algorithm
def genetic_algorithm(population_size=5, num_generations=5, pmut=0.05):
    # Define the hyperparameter space
    param_choices = {
        'epochs': lambda: random.randint(50, 250),
        'lr': lambda: random.choice([1e-5, 1e-2, 5e-5, 1e-4, 1e-3]),
        'batch_size': lambda: random.choice([16, 32, 64, 128]),
        'npass': lambda: random.randint(1, 5),
        'nblocks': lambda: random.randint(1, 5)
    }

    # Create initial population
    def create_individual():
        return {k: v() for k, v in param_choices.items()}

    population = [create_individual() for _ in range(population_size)]

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        start_generation = checkpoint['generation']
        start_individual = checkpoint['individual']
        population = [create_individual() for _ in range(population_size)] #reinitilize population
        population[start_individual] = checkpoint['params'] # set current individual back to checkpointed
        best_mae = checkpoint['mae']

    else:
        start_generation = 0
        start_individual = 0
        best_mae = float('inf')


    with open('optimization_results_GA.txt', 'a') as f:
        for generation in range(start_generation, num_generations):
            for individual_idx in range(start_individual, population_size):

                log_results(f"Generation {generation + 1} Individual {individual_idx + 1}:", f)

                # Evaluate the individual
                params = population[individual_idx]
                log_results(f"Hyperparameters:", f)
                log_results(f"Epochs: {params['epochs']}", f)
                log_results(f"Learning rate: {params['lr']:.6f}", f)
                log_results(f"Batch size: {params['batch_size']}", f)
                log_results(f"Passes: {params['npass']}", f)
                log_results(f"Blocks: {params['nblocks']}\n", f)

                model = initialize_model(params['lr'], params['npass'], params['nblocks'])

                model.train(
                    train_structures,
                    train_targets_normalized,
                    validation_structures=valid_structures,
                    validation_targets=valid_targets_normalized,
                    epochs=params['epochs'],
                    batch_size=params['batch_size']
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

                if overall_mae < best_mae:
                    best_mae = overall_mae
                    best_params = params
                    log_results("New best model found!\n", f)

                # Save checkpoint
                save_checkpoint(generation, individual_idx+1, params, best_mae)

            # Reset individual index for next generation
            start_individual = 0

            # Selection (select top individuals - here, simplest is just to sort)
            ranked_population = sorted(population, key=lambda ind: evaluate_individual(ind, valid_structures, valid_targets, scaler)[0])
            population = ranked_population[:population_size // 2] #take top half
            while len(population) < population_size:
                population.append(create_individual()) # pad with random individuals

            # Crossover (single point crossover - very basic)
            offspring = []
            for i in range(0, population_size, 2):
                parent1 = population[i % len(population)]
                parent2 = population[(i + 1) % len(population)]

                crossover_point = random.choice(list(param_choices.keys()))
                child1 = parent1.copy()
                child2 = parent2.copy()

                # Swap parameters after the crossover point
                keys = list(param_choices.keys())
                crossover_index = keys.index(crossover_point)
                for j in range(crossover_index, len(keys)):
                    key = keys[j]
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
                offspring.extend([child1, child2])
            population = offspring

            # Mutation
            for individual in population:
                if random.random() < pmut:
                    param_to_mutate = random.choice(list(param_choices.keys()))
                    individual[param_to_mutate] = param_choices[param_to_mutate]()  # Generate a new random value

    return best_params


def evaluate_individual(params, structures, targets, scaler):
    """Helper function to evaluate an individual (set of hyperparameters).  Returns MAE."""
    model = initialize_model(params['lr'], params['npass'], params['nblocks'])
    model.train(
        train_structures,
        train_targets_normalized,
        validation_structures=structures,
        validation_targets=targets,
        epochs=params['epochs'],
        batch_size=params['batch_size']
    )
    mae, _, _ = evaluate_model(model, structures, targets)
    overall_mae = np.nanmean(mae)
    return overall_mae, params # return the mae and the parameters


# Main execution
start_time = time.time()
best_params = genetic_algorithm()
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

with open('optimization_results_GA.txt', 'a') as f:
    log_results("\nFinal Model Evaluation on Test Set:", f)
    for i, prop in enumerate(['band_gap', 'formation_energy_per_atom', 'energy_above_hull']):
        log_results(f"MAE for {prop}: {test_mae[i]:.6f}", f)
        log_results(f"R2 for {prop}: {test_r2[i]:.6f}", f)
        log_results(f"Pearson for {prop}: {test_pearson[i]:.6f}", f)
    log_results(f"Average Test MAE: {overall_test_mae:.6f}", f)
    log_results(f"Average Test R2: {overall_test_r2:.6f}", f)
    log_results(f"Average Test Pearson: {overall_test_pearson:.6f}", f)
    log_results(f"Total optimization time: {end_time - start_time:.1f} seconds", f)
