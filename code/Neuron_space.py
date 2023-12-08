import Coordinates
import Axon

import random
import numpy as np



size = 100
class NeuronSpace():
    def __init__(self, Visualization):
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
        else:
            name = n.name + "," + i.name
        if name not in self.Axon_dict.keys():
            axon = Axon.Axon(i, n, name=name, base_space=self)
            self.Axon_dict[name] = axon
            i.connections.append(axon)
            n.connections.append(axon)
            return axon

    def find_x_nearest(self, Neuron, setB, connection_limit=8, x=5): # finds x nearest Neurons of setB to Neuron
        distdict={}
        for i in setB:
            if i != Neuron and len(i.connections) < connection_limit and sum([(type(c.other_side(i)) == Neuron.input_Neuron or c.other_side(i).output) for c in i.connections]) == 0:
                # check if neuron is perceptive and if i already connected to perceptive
                # this should ensure that one perceptive neuron does not connect to a processing neuron thats already connected to a perceptive neuron
                if type(Neuron) == Neuron.input_Neuron:
                    input_connections = [(type(connections_of_i.other_side(connections_of_i)) == Neuron.input_Neuron) for connections_of_i in i.connections]
                    if sum(input_connections) == 0:
                        distdict[Coordinates.distance_finder(Neuron.coordinates, i.coordinates)] = i
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