import numpy as np
import random
random.seed(1)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import Coordinates
import Perceptive_neuron, Processing_neuron, Interaction_neuron
import Axon

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
            axon = Axon.Axon(i, n, name=name, size=size, base_space=self, threshold=0.5*random.uniform(0.7, 1.3))
            self.Axon_dict[name] = axon
            i.connections.append(axon)
            n.connections.append(axon)
            return axon

    def find_x_nearest(self, Neuron, setB, connection_limit=8, x=5): # finds x nearest Neurons of setB to Neuron
        distdict={}
        for i in setB:
            if i != Neuron and len(i.connections) < connection_limit and sum([(type(c.other_side(i)) == Perceptive_neuron.PerceptiveNeuron or type(c.other_side(i)) == Interaction_neuron.InteractionNeuron) for c in i.connections]) == 0:
                # check if neuron is perceptive and if i already connected to perceptive
                # this should ensure that one perceptive neuron does not connect to a processing neuron thats already connected to a perceptive neuron
                if type(Neuron) == Perceptive_neuron.PerceptiveNeuron:
                    perceptive_connections = [(type(connections_of_i.other_side(connections_of_i)) == Perceptive_neuron.PerceptiveNeuron) for connections_of_i in i.connections]
                    if sum(perceptive_connections) == 0:
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
        for key in self.neuron_dot_dict:
            value = self.neuron_dot_dict[key]
            if value[1].active:
                value[0].set_color("red")
            else:
                value[0].set_color("grey")
            value[0].set_sizes([50 * value[1].signal_modification])

        for key in self.axon_line_dict:
            value = self.axon_line_dict[key]
            if value[1].active:
                value[0][0].set_color("red")
            else:
                value[0][0].set_color("grey")
        for n in self.grown_axons:
            value = self.axon_line_dict[n]
            value[0][0].set_color("green")
        if len(self.grown_axons) > 0:
            print("strengthened ", len(self.grown_axons), " axons")
        for n in self.new_axons:
            value = self.axon_line_dict[n.name]
            value[0][0].set_color("purple")
        if len(self.new_axons) > 0:
            print("grew ", len(self.new_axons), " axons")
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

    def Hebbian(self):

        for n in self.active_neurons:
            if n in self.PNeuron_dict.keys():
                n = self.PNeuron_dict[n]
                for z in self.active_neurons:
                    if z in self.PNeuron_dict.keys():
                        z = self.PNeuron_dict[z]
                        if n != z:
                            if z not in [o.other_side(own=n) for o in n.connections] and z.name not in n.fire_together.keys():
                                n.fire_together[z.name] = 2
                            if z.name in n.fire_together.keys():
                                if n.fire_together[z.name] > 30:
                                    a = self.create_Axon(n, z)
                                    if a != None:
                                        self.new_axons.append(a)
                                        #print("Axon grown!")
                                        if self.Visualization:
                                            self.axon_line_dict[a.name] = [
                                                (self.ax.plot3D([a.neuron1.coordinates.x, a.neuron2.coordinates.x],
                                                                [a.neuron1.coordinates.y, a.neuron2.coordinates.y],
                                                                [a.neuron1.coordinates.z, a.neuron2.coordinates.z], linewidth=1,
                                                                c='grey')), a]
                                    n.fire_together = {}
                                elif z.name in n.fire_together.keys():
                                    n.fire_together[z.name] += 3


    def run(self):
        if self.learn:
            for i in self.Pset:
                for f in list(i.fire_together):
                    i.fire_together[f] -= 1
                    if i.fire_together[f] == 0:
                        i.fire_together.pop(f)
            self.Hebbian()


        aavalues = list(self.active_axons.values())
        for i in aavalues:
            i.step()
        anvalues = list(self.active_neurons.values())
        for i in anvalues:
            i.step()

        #hier gehts weiter


    def spawn_neurons_axons(self):

        # TODO
        # choose coordinates for perceptive neurons, set V
        # these should face one plane of the neuron space
        V = []
        mean = [0, 0]
        cov = [[100, 100], [100, 0]]
        # np.random.multivariate_normal(mean, cov, 1).T

        for v in np.arange(1):  # how many neurons do we want
            V = self.ordered_input_neurons(height = 8, width = 8, plane_end=-(size/2))
            #y, z = self.new_positions_circular_coordinates()
            #V.append(Coordinates.Coordinate(-(size/2), y, z))

        # choose cluster of coordinates in the middle of the neuron space for processing neurons, set P
        P = []
        for p in np.arange(200):
            x, y, z = self.new_positions_spherical_coordinates()
            P.append(Coordinates.Coordinate(x, y, z))

        # choose cluster of coordinates on plane, opposite side to V, set I
        # that only connect to processing neurons
        I = []
        for i in np.arange(1):  # how many neurons do we want
            I = self.ordered_output_neurons(height=10, width=1, plane_end=size/2)
            #y, z = self.new_positions_circular_coordinates()
            #np.random.multivariate_normal(mean, cov, 1).T
            #I.append(Coordinates.Coordinate(size/2, y, z))


        # Neuron generation

        # spawn a bunch of Perceptive neurons on coordinate set V
        self.Vset = []
        for v in V:
            self.Vset.append(Perceptive_neuron.PerceptiveNeuron(v, [], 1.2, base_space = self))
        # spawn a bunch of Processing neurons on coordinate set P
        self.Pset = []
        for p in P:
            self.Pset.append(Processing_neuron.ProcessingNeuron(p, [], 1, base_space = self))
        # spawn a bunch of Interaction neurons on coordinate set I
        self.Iset = []
        for idx, i in enumerate(I):
            self.Iset.append(Interaction_neuron.InteractionNeuron(i, [], 1, base_space = self, id = str(idx)))


        self.Axon_dict = {}

        # axons generation from Perception to 1 nearest neurons in processing neuron set
        # perceptives should only connect to a processing neuron that is not directly connected to another perceptive
        for v in self.Vset:
            Ns = self.find_x_nearest(v,self.Pset, connection_limit=25, x=1)
            for n in Ns:
                self.create_Axon(v, n)

        # axons generation from Interaction to 3 nearest neurons in processing neuron set
        for i in self.Iset:
            Ns = self.find_x_nearest(i,self.Pset, connection_limit=25, x=1)
            for n in Ns:
                self.create_Axon(i, n)

        # axons generation from Processing to 3 nearest neurons in processing neuron set
        for p in self.Pset:
            Ns = self.find_x_nearest(p,self.Pset,connection_limit=20, x=5)
            for n in Ns:
                self.create_Axon(p, n)

        self.PNeuron_dict = {}
        for p in self.Pset:
            self.PNeuron_dict[p.name] = p

        self.INeuron_dict = {}
        for i in self.Iset:
            self.INeuron_dict[i.name] = i

        self.VNeuron_dict = {}
        for v in self.Vset:
            self.VNeuron_dict[v.name] = v

        self.Neuron_dict = self.PNeuron_dict.copy()
        self.Neuron_dict.update(self.INeuron_dict)
        self.Neuron_dict.update(self.VNeuron_dict)

        self.grown_axons = []
        self.new_axons = []
        if self.Visualization:
            self.start_vis()
            #self.draw_brain(active_axons={})

    def feed_image_unsupervised(self, datapoint, target, learn):
        self.learn = learn
        self.stop = False # this becomes true once an interactive neuron gets activated
        if self.learn:
            print("Learning")
        else:
            print("not learning")
        self.generate = False
        self.active_neurons = {}
        self.active_axons = {}

        for v, idx in zip(self.Vset, np.arange(len(self.Vset))):
            v.activation(source="input", signal = datapoint[idx])
            # TODO
            # das wieder rein nehmen, damit vorhersage mit output abgeglichen werden kann
            #self.Iset[target].activation(1)
        training_runs = 0
#        self.run()
#        self.ticks += 1
        if self.Visualization:
            self.draw_brain(self.active_axons)

        # if no input neurons or processing neurons or axons are active
        while len(self.active_neurons.keys()) + len(self.active_axons.keys()) > 0 or training_runs==0:
            self.grown_axons = []
            self.new_axons = []
            self.run()
            self.ticks += 1
            print("Tick: ", self.ticks)
            if self.Visualization:
                self.draw_brain(self.active_axons)
            training_runs += 1
            if self.stop:
                break
            # solange irgend ein neuron noch an is:
            # die neuronen sind länger an, senden aber nur am anfang ein signal an die axone

        if self.learn:
            print("Decay axons")
            self.to_remove = []
            for all_axons in self.Axon_dict.values():
                all_axons.forget()

            # only one is deleted to avoid mass extinction
            if len(self.to_remove) >= 1:
                self.Axon_dict.pop(self.to_remove[0].name)
                if self.Visualization:
                    self.axon_line_dict.pop(self.to_remove[0].name)
                self.to_remove[0].neuron1.connections.remove(self.to_remove[0])
                self.to_remove[0].neuron2.connections.remove(self.to_remove[0])
                del self.to_remove[0]

            if len(self.to_remove) > 0:

                print(len(self.to_remove), " axons at max threshold")


        # clean up for next run
        self.active_neurons = {}
        self.active_axons = {}
        self.stop = False
        for ad in self.Axon_dict:
            self.Axon_dict[ad].last_signal = self.ticks-self.Axon_dict[ad].cooldown-1
            self.Axon_dict[ad].reset_for_next_run()
        for Nd in self.Neuron_dict:
            self.Neuron_dict[Nd].reset_for_next_run()

        print("Picture processed")
        # self.start_vis()
        # self.draw_brain([])

    def generate_image(self, datapoint, target):
        self.generate = True
        self.active_neurons = {}
        self.active_axons = {}

        self.Iset[target].activation("input", 1)

        self.run()
        self.ticks += 1
        if self.Visualization:
            self.draw_brain(self.active_axons)
        while len(self.active_neurons.keys()) + len(self.active_axons.keys()) != 0:
            self.run()
            for all_axons in self.Axon_dict.values():
                all_axons.forget()
            self.ticks += 1
            print("Tick: ", self.ticks)
            if self.Visualization:
                self.draw_brain(self.active_axons)
            # solange irgend ein neuron noch an is:
            # die neuronen sind länger an, senden aber nur am anfang ein signal an die axone
        print("Bild generiert")


neuronspace = NeuronSpace(Visualization = False)
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
min_max_scaler = preprocessing.MinMaxScaler()
#training data
df = pd.DataFrame(datasets.load_digits().data)
#fit and transform the training data and use them for the model training
df1 = min_max_scaler.fit_transform(df)
digits = datasets.load_digits()
images = 0

print("Sensing images \n"
      "Symptoms: \n"
      "-Quick decay of axons \n"
      "-Activation function is the binary step function to simulate synapse transmitting signals \n"
      "-Not emergent")

for d,t in zip(df1, digits.target):
    #neuronspace.generate_image(d, t)
    print("durchgang ", images, " start tick: ", neuronspace.ticks)
    neuronspace.feed_image_unsupervised(d, t, learn = True)
    images += 1
    if images == 20:
        print("stop")
    if images == 10:
        print("stop")
        neuronspace.Visualization = True
        plt.ion()
        neuronspace.start_vis()


print("check")