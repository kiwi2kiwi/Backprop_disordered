import Coordinates
import Axon
import matplotlib.pyplot as plt

import Neuron
import random
import numpy as np
np.random.seed(1)
random.seed(1)

size = 100
class NeuronSpace():
    def __init__(self, Visualization = True, fast = False, neuron_number = 10):
        super(NeuronSpace, self).__init__()
        self.fast = fast
        self.Visualization = Visualization
        if self.fast:
            self.Visualization = False
        self.neuron_number = neuron_number

    def new_positions_spherical_coordinates(self):
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        u = random.uniform(0, 1)

        theta = np.arccos(costheta)
        r = ((size-10) / 2) * np.sqrt(u)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return (round(x, 3), round(y, 3), round(z, 3))

    def new_positions_circular_coordinates(self):
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        u = random.uniform(0, 1)

        size = 100
        theta = np.arccos(costheta)
        r = (size / 2) * np.sqrt(u)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        #z = r * np.cos(theta)
        return (round(x,3), round(y,3))

    def ordered_input_neurons(self, height, width, plane_end):
        global size
        V = []
        area = size - 20
        y_distance = area / height
        z_distance = area / width
        Y = np.arange(-(size / 2) + 10, (size / 2) - 10, y_distance)
        Z = np.arange(-(size / 2) + 10, (size / 2) - 10, z_distance)
        for y in Y:
            for z in Z:
                V.append(Coordinates.Coordinate(plane_end, y, z))
        return V

    def ordered_output_neurons(self, height, width, plane_end):
        global size
        V = []
        area = size - 20
        y_distance = area / height
        z_distance = area / width
        Y = np.arange(-(size/2)+10,(size/2)-10,y_distance)
        Z = np.arange(-(size/2)+10,(size/2)-10,z_distance)
        for y in Y:
            for z in Z:
                V.append(Coordinates.Coordinate(plane_end, y, 0))
        return V

    def create_Axon(self, i, n):
        is_n_parent = True
        if i.coordinate.x < n.coordinate.x:
            is_n_parent = False
        elif i.coordinate.x == n.coordinate.x:
            if i.coordinate.y < n.coordinate.y:
                is_n_parent = False
            elif i.coordinate.y == n.coordinate.y:
                if i.coordinate.z < n.coordinate.z:
                    is_n_parent = False

        if is_n_parent:
            name = n.name + i.name
            if name not in self.Axon_dict.keys():
                weight = round(random.uniform(0, 1), 2)
                axon = Axon.Axon(n, i, name=name, base_space=self, weight=weight, new_weights=[])
                self.Axon_dict[name] = axon
                i.parent_connections[n.name] = axon
                n.children_connections[i.name] = axon
                return axon
        else:
            name = i.name + n.name
            if name not in self.Axon_dict.keys():
                weight = round(random.uniform(0, 1), 2)
                axon = Axon.Axon(i, n, name=name, base_space=self, weight=weight, new_weights=[])
                self.Axon_dict[name] = axon
                i.children_connections[n.name] = axon
                n.parent_connections[i.name] = axon
                return axon

    def find_x_nearest(self, neuron, setB, connection_limit=8, x=5): # finds x nearest Neurons of setB to Neuron
        perceptive_connection_limit = 4
        distdict={}
        for i in setB:
            if i != neuron and len(i.parent_connections) < connection_limit: # and sum([(type(c.other_side(i)) == Neuron.Input_Neuron or c.other_side(i).output) for c in i.parent_connections]) == 0:
                # check if neuron is perceptive and if i already connected to perceptive
                # this should ensure that one perceptive neuron does not connect to a processing neuron thats already connected to a perceptive neuron
                if type(neuron) == Neuron.Input_Neuron:
                    if len(i.parent_connections) < perceptive_connection_limit:
                        distdict[Coordinates.distance_finder(neuron.coordinate, i.coordinate)] = i
                    # Debug output
#                    else:
#                        print("prevented perceptives connecting to same neuron")
                else:
                    distdict[Coordinates.distance_finder(neuron.coordinate, i.coordinate)] = i
        srtd = sorted(distdict.items())
        return [i[1] for i in srtd[:x]]

    def draw_brain(self):
        bias_list = []
        weight_list = []
        for neuron in self.neurons:
            bias_list.append(neuron.bias)
            if type(neuron) != Neuron.Input_Neuron:
                for p_axon in neuron.parent_connections.values():
                    weight_list.append(p_axon.get_weight())

        cmap = plt.get_cmap('YlOrRd')  # 'cool')
        norm_bias = plt.Normalize(min(bias_list), max(bias_list))
        norm_weight = plt.Normalize(min(weight_list), max(weight_list))

        # visualize the neurons
        for key in self.neuron_dot_dict:
            value = self.neuron_dot_dict[key]
            color = cmap(norm_bias(value[1].bias))
            value[0].set_color(color)

        # visualize the axons strength
        for key in self.axon_line_dict:
            value = self.axon_line_dict[key]
            color = cmap(norm_weight(value[1].get_weight()))

            value[0][0].set_color(color)

        #self.fig.savefig('..//Bilder//temp'+str(self.ticks)+'.png', dpi=self.fig.dpi)

    def print_states(self):
        for name in self.Neuron_dict:
            neuron = self.Neuron_dict[name]
            print(neuron.name , " bias: ", neuron.bias)

        for name in self.Axon_dict:
            axon = self.Axon_dict[name]
            print(axon.name , " weight: ", axon.weight)

    def start_vis(self):
        plt.ion()
        self.neuron_dot_dict = {}  # name: (neuron, punkt auf plot)
        self.axon_line_dict = {}  # name: (axon, linie auf plot)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-(size / 2), size / 2)
        self.ax.set_ylim(-(size / 2), size / 2)
        self.ax.set_zlim(-(size / 2), size / 2)
        for i in self.input_set:  # plot perceptive neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinate.x, i.coordinate.y, i.coordinate.z, c="grey",
                                                             s=10)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for i in self.hidden_set:  # plot processing neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinate.x, i.coordinate.y, i.coordinate.z, c="grey",
                                                             s=10)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for i in self.output_set:  # plot interaction neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinate.x, i.coordinate.y, i.coordinate.z, c="grey",
                                                             s=10)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for a in self.Axon_dict.values():
            self.axon_line_dict[a.name] = [(self.ax.plot3D([a.parent.coordinate.x, a.child.coordinate.x],
                                                           [a.parent.coordinate.y, a.child.coordinate.y],
                                                           [a.parent.coordinate.z, a.child.coordinate.z], linewidth=1,
                                                           c='grey')), a]

        self.grown_axons = []

    def spawn_neurons_axons(self, input_number, output_number):

        mean = [0, 0]
        cov = [[100, 100], [100, 0]]
        # np.random.multivariate_normal(mean, cov, 1).T

        I = []
        for i in np.arange(input_number):  # how many neurons do we want
            I = self.ordered_input_neurons(height = np.sqrt(input_number), width = np.sqrt(input_number), plane_end=-(size/2))
            #y, z = self.new_positions_circular_coordinates()
            #V.append(Coordinates.Coordinate(-(size/2), y, z))

        # choose cluster of coordinates in the middle of the neuron space for processing neurons, set P
        P = []
        for p in np.arange(self.neuron_number):
            x, y, z = self.new_positions_spherical_coordinates()
            P.append(Coordinates.Coordinate(x, y, z))

        # choose cluster of coordinates on plane, opposite side to V, set I
        # that only connect to processing neurons
        O = []
        for o in np.arange(output_number):  # how many neurons do we want
            O = self.ordered_output_neurons(height=output_number, width=1, plane_end=size/2)
            #y, z = self.new_positions_circular_coordinates()
            #np.random.multivariate_normal(mean, cov, 1).T
            #I.append(Coordinates.Coordinate(size/2, y, z))


        # Neuron generation

        # spawn a bunch of output neurons on coordinate set output_set
        self.output_set = []
        for o in O:
            self.output_set.append(Neuron.Neuron(o, self, True, bias=0))
        # spawn a bunch of Processing neurons on coordinate set P
        self.hidden_set = []
        for h in P:
            self.hidden_set.append(Neuron.Neuron(h, base_space = self, bias=0))
        # spawn a bunch of input neurons on coordinate set I
        self.input_set = []
        for idx, i in enumerate(I):
            self.input_set.append(Neuron.Input_Neuron(i, base_space = self))


        self.Axon_dict = {}

        # axons generation from Perception to 1 nearest neurons in processing neuron set
        # perceptives should only connect to a processing neuron that is not directly connected to another perceptive
        for o in self.output_set:
            Ns = self.find_x_nearest(o, self.hidden_set, connection_limit=20, x=10)
            for n in Ns:
                self.create_Axon(o, n)

        # axons generation from Interaction to 3 nearest neurons in processing neuron set
        for i in self.input_set:
            Ns = self.find_x_nearest(i, self.hidden_set, connection_limit=9, x=2)
            for n in Ns:
                self.create_Axon(i, n)

        # axons generation from Processing to 3 nearest neurons in processing neuron set
        for p in self.hidden_set:
            Ns = self.find_x_nearest(p, self.hidden_set, connection_limit=30, x=10)
            for n in Ns:
                self.create_Axon(p, n)

        self.hidden_neuron_dict = {}
        for h in self.hidden_set:
            self.hidden_neuron_dict[h.name] = h

        self.input_neuron_dict = {}
        for i in self.input_set:
            self.input_neuron_dict[i.name] = i

        self.output_neuron_dict = {}
        for o in self.output_set:
            self.output_neuron_dict[o.name] = o

        self.neurons = self.input_set + self.hidden_set + self.output_set

        self.Neuron_dict = self.hidden_neuron_dict.copy()
        self.Neuron_dict.update(self.input_neuron_dict)
        self.Neuron_dict.update(self.output_neuron_dict)

        self.grown_axons = []
        self.new_axons = []
        if self.Visualization:
            self.start_vis()
            plt.show()
            print("done starting the visualization!")
            #self.draw_brain(active_axons={})



