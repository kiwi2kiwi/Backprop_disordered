import Coordinates
import Axon
import matplotlib.pyplot as plt

import Neuron
import random
import numpy as np


size = 100
class NeuronSpace():
    def __init__(self, Visualization = True):
        super(NeuronSpace, self).__init__()
        self.iter = 0
        self.ticks = 0
        self.generate = False
        self.Visualization = Visualization
        self.spawn_neurons_axons()

    def new_positions_spherical_coordinates(self):
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        u = random.uniform(0, 1)

        theta = np.arccos(costheta)
        r = ((size-10) / 2) * np.sqrt(u)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return (x, y, z)

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
        return (x, y)

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
        in_name_v_first = True
        if i.coordinates.x < n.coordinates.x:
            in_name_v_first = False
        elif i.coordinates.x == n.coordinates.x:
            if i.coordinates.y < n.coordinates.y:
                in_name_v_first = False
            elif i.coordinates.y == n.coordinates.y:
                if i.coordinates.z < n.coordinates.z:
                    in_name_v_first = False
        name = ""
        if in_name_v_first:
            name = i.name + "," + n.name
            if name not in self.Axon_dict.keys():
                axon = Axon.Axon(i, n, name=name, base_space=self)
                self.Axon_dict[name] = axon
                i.parent_connections[n] = [n, 100, []]
#                n.children_connections[i] = [i, 100, []]
                return axon
        else:
            name = n.name + "," + i.name
            if name not in self.Axon_dict.keys():
                axon = Axon.Axon(i, n, name=name, base_space=self)
                self.Axon_dict[name] = axon
                #i.children_connections[n] = [n, 100, []]
                n.parent_connections[i] = [i, 100, []]
                return axon


    def find_x_nearest(self, Neuron, setB, connection_limit=8, x=5): # finds x nearest Neurons of setB to Neuron
        distdict={}
        for i in setB:
            if i != Neuron and len(i.parent_connections) < connection_limit and sum([(type(c.other_side(i)) == Neuron.Input_Neuron or c.other_side(i).output) for c in i.parent_connections]) == 0:
                # check if neuron is perceptive and if i already connected to perceptive
                # this should ensure that one perceptive neuron does not connect to a processing neuron thats already connected to a perceptive neuron
                if type(Neuron) == Neuron.Input_Neuron:
                    input_connections = [(type(connections_of_i.other_side(connections_of_i)) == Neuron.Input_Neuron) for connections_of_i in i.connection]
                    if sum(input_connections) == 0:
                        distdict[Coordinates.distance_finder(Neuron.coordinate, i.coordinate)] = i
                    # Debug output
#                    else:
#                        print("prevented perceptives connecting to same neuron")
                else:
                    distdict[Coordinates.distance_finder(Neuron.coordinates, i.coordinates)] = i
        srtd = sorted(distdict.items())
        return [i[1] for i in srtd[:x]]

    def draw_brain(self, active_axons):
        # visualize the neurons
        # TODO change this to display neuron activation
        for key in self.neuron_dot_dict:
            value = self.neuron_dot_dict[key]
            if value[1].active:
                value[0].set_color("red")
            else:
                value[0].set_color("grey")
            value[0].set_sizes([50 * value[1].signal_modification])

        # TODO change this to display weight value
        for key in self.axon_line_dict:
            value = self.axon_line_dict[key]
            if value[1].active:
                value[0][0].set_color("red")
            else:
                value[0][0].set_color("grey")
        self.fig.savefig('..//Bilder//temp'+str(self.ticks)+'.png', dpi=self.fig.dpi)
        self.grown_axons=[]
        self.new_axons = []


    def start_vis(self):
        #plt.ion()
        self.neuron_dot_dict = {}  # name: (neuron, punkt auf plot)
        self.axon_line_dict = {}  # name: (axon, linie auf plot)
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-(size / 2), size / 2)
        self.ax.set_ylim(-(size / 2), size / 2)
        self.ax.set_zlim(-(size / 2), size / 2)
        for i in self.Vset:  # plot perceptive neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinates.x, i.coordinates.y, i.coordinates.z, c="grey",
                                                             s=10 * i.signal_modification)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for i in self.Pset:  # plot processing neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinates.x, i.coordinates.y, i.coordinates.z, c="grey",
                                                             s=10 * i.signal_modification)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for i in self.Iset:  # plot interaction neurons
            self.neuron_dot_dict[i.name] = [(self.ax.scatter(i.coordinates.x, i.coordinates.y, i.coordinates.z, c="grey",
                                                             s=10 * i.signal_modification)), i]
        #    for c in i.connections:
        #        ax.plot3D([c.neuron1.coordinats.x, c.neuron2.coordinats.x], [c.neuron1.coordinats.y, c.neuron2.coordinats.y], [c.neuron1.coordinats.z, c.neuron2.coordinats.z], 'b')

        for a in self.Axon_dict.values():
            self.axon_line_dict[a.name] = [(self.ax.plot3D([a.neuron1.coordinates.x, a.neuron2.coordinates.x],
                                                           [a.neuron1.coordinates.y, a.neuron2.coordinates.y],
                                                           [a.neuron1.coordinates.z, a.neuron2.coordinates.z], linewidth=1,
                                                           c='grey')), a]

        self.grown_axons = []

    def spawn_neurons_axons(self):

        mean = [0, 0]
        cov = [[100, 100], [100, 0]]
        # np.random.multivariate_normal(mean, cov, 1).T

        I = []
        for i in np.arange(1):  # how many neurons do we want
            I = self.ordered_input_neurons(height = 8, width = 8, plane_end=-(size/2))
            #y, z = self.new_positions_circular_coordinates()
            #V.append(Coordinates.Coordinate(-(size/2), y, z))

        # choose cluster of coordinates in the middle of the neuron space for processing neurons, set P
        P = []
        for p in np.arange(20):
            x, y, z = self.new_positions_spherical_coordinates()
            P.append(Coordinates.Coordinate(x, y, z))

        # choose cluster of coordinates on plane, opposite side to V, set I
        # that only connect to processing neurons
        O = []
        for o in np.arange(1):  # how many neurons do we want
            O = self.ordered_output_neurons(height=10, width=1, plane_end=size/2)
            #y, z = self.new_positions_circular_coordinates()
            #np.random.multivariate_normal(mean, cov, 1).T
            #I.append(Coordinates.Coordinate(size/2, y, z))


        # Neuron generation

        # spawn a bunch of output neurons on coordinate set output_set
        self.output_set = []
        for o in O:
            self.output_set.append(Neuron.Neuron(o, self, True))
        # spawn a bunch of Processing neurons on coordinate set P
        self.hidden_set = []
        for h in P:
            self.hidden_set.append(Neuron.Neuron(h, base_space = self))
        # spawn a bunch of input neurons on coordinate set I
        self.input_set = []
        for idx, i in enumerate(I):
            self.input_set.append(Neuron.Input_Neuron(i, base_space = self))


        self.Axon_dict = {}

        # axons generation from Perception to 1 nearest neurons in processing neuron set
        # perceptives should only connect to a processing neuron that is not directly connected to another perceptive
        for o in self.output_set:
            Ns = self.find_x_nearest(o, self.hidden_set, connection_limit=25, x=1)
            for n in Ns:
                self.create_Axon(o, n)

        # axons generation from Interaction to 3 nearest neurons in processing neuron set
        for i in self.input_set:
            Ns = self.find_x_nearest(i, self.hidden_set, connection_limit=25, x=1)
            for n in Ns:
                self.create_Axon(i, n)

        # axons generation from Processing to 3 nearest neurons in processing neuron set
        for p in self.hidden_set:
            Ns = self.find_x_nearest(p, self.hidden_set, connection_limit=20, x=5)
            for n in Ns:
                self.create_Axon(p, n)

        self.hidden_neuron_dict = {}
        for h in self.hidden_set:
            self.hidden_neuron_dict[p.name] = h

        self.input_neuron_dict = {}
        for i in self.input_set:
            self.input_neuron_dict[i.name] = i

        self.output_neuron_dict = {}
        for o in self.output_set:
            self.output_neuron_dict[o.name] = o

        self.Neuron_dict = self.hidden_neuron_dict.copy()
        self.Neuron_dict.update(self.input_neuron_dict)
        self.Neuron_dict.update(self.output_neuron_dict)

        self.grown_axons = []
        self.new_axons = []
        if self.Visualization:
            self.start_vis()
            print("done")
            #self.draw_brain(active_axons={})

n = NeuronSpace()
n.spawn_neurons_axons()