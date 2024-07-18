import qiskit
import qiskit_ibm_runtime
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import numpy as np 
import math as math
import os


dict1 = {}
cosstdict = {}
lr_cost = {}
target_state_global = [1,0]
taget_states = [[1,1],[0,1],[1,0],[1/np.sqrt(2),1/np.sqrt(2)]]
def create_qc(parameters):
    circuit = QuantumCircuit(1, 0)
    circuit.rx(parameters[0], 0)
    return circuit

def cost_function(statevector):
    # Example cost function (you can customize this based on your task)
    target_state = target_state_global  # Target state, e.g., |0⟩
    cost = sum(abs(target - actual) ** 2 for target, actual in zip(target_state, statevector.data))
    return cost

def qgd_steps(params, learning_rate,ep):
    qc = create_qc(params)

    # Simulate qc using statevector_simulator
    simulator = Aer.get_backend('statevector_simulator')
    compiled_circuit = transpile(qc, simulator)
    end_result = simulator.run(compiled_circuit).result()
    statevector = end_result.get_statevector()

    # Calculate the cost
    current_cost = cost_function(statevector)
    
    # Numerical gradient (simple example)
    epsilon = ep
    qc_plus = create_qc([params[0] + epsilon])
    compiled_circuit_plus = transpile(qc_plus, simulator)
    result_plus = simulator.run(compiled_circuit_plus).result()
    statevector_plus = result_plus.get_statevector()
    numerical_gradient = (cost_function(statevector_plus) - current_cost) / epsilon

    # Update parameter using gradient descent
    updated_params = [params[0] - learning_rate * numerical_gradient]

    return updated_params



def Quantum_GD(initial_parameters, learning_rate, iterations,episolon):
    params = initial_parameters
    i = 0
    
    for i in range(iterations):
        params = qgd_steps(params, learning_rate, episolon)
        i += 1
        if i == 2:
            episolon /= 2
            i = 0

    
    return params


def run_qc(initial_parameters):
    for params in initial_parameters:
        qcy = create_qc(params)

        # Simulate qc using statevector_simulator
        simulator2 = Aer.get_backend('statevector_simulator')
        compiled_circuit = transpile(qcy, simulator2)
        end_result = simulator2.run(compiled_circuit).result()
        statevector = end_result.get_statevector()

        # Format the state vector for display
        formatted_statevector = [f"{np.round(x.real, 3) if x.imag == 0 else ''}{ np.round(x.imag, 3) if x.real == 0 else ''}" for x in statevector.data]
        #add J to make it an actual complex number

        print(f"Initial Parameters: {params}")
        print(f"Statevector Result: {formatted_statevector[0]}, {formatted_statevector[1]}")
        dict1[params[0]]= formatted_statevector
        print("-" * 40)



def visualize_circuit_evolution(initial_params, learning_rate, iterations, epsilon, output_folder):
    params = initial_params

    if not os.path.exists(output_folder):
         os.makedirs(output_folder)


    for i in range(iterations):
        qc = create_qc(params)

        # Simulate qc using statevector_simulator
        simulator = Aer.get_backend('statevector_simulator')
        compiled_circuit = transpile(qc, simulator)
        end_result = simulator.run(compiled_circuit).result()
        statevector = end_result.get_statevector()

        # Visualize the circuit
        plt.figure(figsize=(12, 4))

        # Plot the circuit
        plt.subplot(1, 2, 1)
        qiskit.visualization.circuit_drawer(compiled_circuit, output='mpl', scale=0.5, style = 'clifford')

        # Plot the state vector
        plt.subplot(1, 2, 2)
        plot_bloch_multivector(statevector)

        plt.suptitle(f'Iteration {i+1}')

        output_filename = os.path.join(output_folder, f"iteration_{i+1}.png")
        plt.savefig(output_filename)
        params = qgd_steps(params, learning_rate, epsilon)
    plt.close()




# Example usage
initial_params = [0.5]
iterations = 8

learning_rates = [0.01,0.05,0.1,0.15,0.2,0.3,0.5,1,1.5,1.7,2,2.2,2.5,2.7,2.85,3,3.2,3.5,4.5,5,6,7,8,9,10]
learning_rates_realistic = [0.01,0.05,0.1,0.15,0.2,0.3,0.5,1,1.5,1.7,2,2.2,2.5,2.7,2.85,3]
#optimal learning rate for this appears to be 2
outsparams = []
'''
input_for_type_of_print = input("Type of print: custom for custom ")
print(input_for_type_of_print)


 if input_for_type_of_print.lower() == "custom".lower:
    #ok
    custom_qx = [float(input("rx"))]
    print("Episolon:",str(custom_qx))
    custom_Learningrate = float(input("Learning rate: "))
    print("learning rate",str(custom_Learningrate))
    run_qc([Quantum_GD(initial_params,custom_Learningrate,iterations,custom_qx)])
'''
for thing in learning_rates_realistic:
    print("learning rate",str(thing))
    run_qc([Quantum_GD(initial_params,thing,iterations,0.01)])
    visualize_circuit_evolution(initial_params,thing, iterations,0.01,output_folder="/Users/morabp27/Desktop/PLT_VISUALIZER_IMAGES")

for parameterstart in dict1:
            cost_for_one = abs((float(dict1[parameterstart][0])- target_state_global[0]))
            cost_for_two = abs(float(dict1[parameterstart][1]) - target_state_global[1])
            total_cost = cost_for_one + cost_for_two
            cosstdict[parameterstart] = total_cost

for params in cosstdict:
     if cosstdict[params] == 0:
          print(f"Congrats!! optimal parameters are {params}!!")

counter = 0 
for thing in cosstdict:
     lr_cost[learning_rates[counter]] = np.round(cosstdict[thing],4)
     counter += 1

learning_rates = list(lr_cost.keys())
costs = list(lr_cost.values())

# Plotting
# plt.plot(learning_rates_realistic, costs, marker='o', linestyle='-', color='b')
# plt.title('Learning Rates vs. Costs')
# plt.xlabel('Learning Rate')
# plt.ylabel('Total Cost')
# plt.grid(True)
# plt.show()