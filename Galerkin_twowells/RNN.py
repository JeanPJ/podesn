__author__ = 'jean'
import numpy as np
import scipy.io as io


def sparsity(M, psi):
    N = np.empty_like(M)
    for linha in range(len(N)):
        for coluna in range(len(N[linha])):
            prob = np.random.rand()
            if prob < psi:
                N[linha][coluna] = 0
            else:
                N[linha][coluna] = 1


    return N*M

def grad_tanh(z):
    
    
    return np.diag(1 - np.tanh(z.flatten())**2)

class EchoStateNetwork:

    def __init__(self,neu,n_in,n_out,
                 gama=0.5,ro=1,psi=0.5,in_scale=0.1,
                 bias_scale=0.5,alfa=10,forget = 1,
                 initial_filename="initial",
                 load_initial = False,save_initial = False,output_feedback = False, 
                 noise_amplitude = 0, 
                 out_scale = 0):
        #All matrixes are initialized under the normal distribution.
        print("initializing reservoir")
        self.neu = neu
        self.n_in = n_in
        self.n_out = n_out
        
        self.psi = psi #the network's sparcity, in 0 to 1 notation
        # Reservoir Weight matrix.
        self.Wrr0 = np.random.normal(0,1,[neu,neu])
        self.Wrr0 = sparsity(self.Wrr0, self.psi)
        # input-reservoir weight matrix
        self.Wir0 = np.random.normal(0,1,[neu,n_in])
        # bias-reservoir weight matrix
        self.Wbr0 = np.random.normal(0,1,[neu,1])

        self.Wor0 = np.random.normal(0,1,[neu,n_out])

        #self.Wbo = np.random.normal(0,1,[n_out,1])
        # reservoir-output weight matrix
        
        self.Wro = np.random.normal(0,1,[n_out,neu+1])
        #self.Wro = np.zeros([n_out,neu])

        self.leakrate = gama #the network's leak rate
        self.ro = ro #the network's desired spectral radius
        self.in_scale = in_scale #the scaling of Wir.
        self.bias_scale = bias_scale #the scaling of Wbr

        # learning rate of the Recursive Least Squares Algorithm
        self.alfa = alfa
        # forget factor of the RLS Algorithm
        self.forget = forget
        self.output_feedback = output_feedback

        #self.a = np.random.normal(0, 1, [neu, 1])
        self.a = np.zeros([neu, 1],dtype=np.float64)
        #save if save is enabled
        if save_initial:
            self.save_initial_fun(initial_filename)

        #load if load is enabled
        if load_initial:
            self.load_initial(initial_filename)

        # the probability of a memeber of the Matrix Wrr being zero is psi.
        self.Wrr = self.Wrr0
        #forcing Wrr to have ro as the maximum eigenvalue
        eigs = np.linalg.eigvals(self.Wrr)
        radius = np.abs(np.max(eigs))
        #normalize matrix
        self.Wrr = self.Wrr/radius
        #set its spectral radius to rho
        self.Wrr *= ro


        #scale tbe matrices
        self.Wbr = bias_scale*self.Wbr0
        self.Wir = in_scale*self.Wir0
        self.Wor = out_scale*self.Wor0
        self.noise = noise_amplitude



        #initial conditions variable forget factor.
        self.sigma_e = 0.001*np.ones(n_out)
        self.sigma_q = 0.001
        self.sigma_v = 0.001*np.ones(n_out)
        self.K_a = 6
        self.K_b = 3*self.K_a

        #covariance matrix
        self.P = np.eye(neu+1)/alfa
        
    def get_n_out(self):
        return self.n_out
    
    def get_n_in(self):
        return self.n_in
    
    def get_current_state(self):
        return self.a
    
    def set_current_state(self,a):
        self.a = a

    def get_wro(self,n=0): #retorna a coluna n do vetor Wro

        return self.Wro[n]

    def training_error(self,ref):

        Ref = np.array(ref,dtype = np.float64)
        if self.n_out > 1:
            Ref = Ref.reshape(len(ref),1)

        e = np.dot(self.Wro,np.vstack((np.atleast_2d(1.0),self.a))) - Ref
        return e

    def train(self,ref):
        #ref e o vetor de todas as sa
        # idas desejados no dado instante de tempo.
        #calcular o vetor de erros
        e = self.training_error(ref)
        #the P equation step by step
        a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
        self.P = self.P/self.forget - np.dot(np.dot(np.dot(self.P,a_wbias),a_wbias.T),self.P)/(self.forget*(self.forget + np.dot(np.dot(a_wbias.T,self.P),a_wbias)))
        #self.sigma_q = (1 - 1 / (self.K_a * self.neu)) * \
        #               self.sigma_q + \
        #               (1 - (1 - 1 / (self.K_a * self.neu))) * \
        #               (np.dot(np.dot(self.a.T, self.P), self.a)) \
        #               ** 2
        for saida in range(self.n_out):

            #self.sigma_e[saida] = (1 - 1/(self.K_a*
            #                       self.neu))*self.sigma_e[saida] + \
            #               (1 - (1 - 1/(self.K_a*self.neu)))*\
            #               e[saida]**2

            #self.sigma_v[saida] = (1 - 1/(self.K_b*self.neu))\
            #               *self.sigma_v[saida] + \
            #               (1 - (1 - 1/(self.K_b*self.neu)))*e[saida]**2
            #self.forget = np.min([(np.sqrt(self.sigma_q) *
            #                       np.sqrt(np.linalg.norm(self.sigma_v)))
            #                       /(10**-8 + abs(np.sqrt(np.linalg.norm(self.sigma_e)) -
            #                       np.sqrt(np.linalg.norm(self.sigma_v)))),0.9999])
            #Transpose respective output view..
            Theta = self.Wro[saida,:]
            Theta = Theta.reshape([self.neu+1,1])


            #error calculation
            Theta = Theta - e[saida]*np.dot(self.P,a_wbias)
            Theta = Theta.reshape([1,self.neu+1])

            self.Wro[saida,:] = Theta




    def update(self,inp,y_in = np.atleast_2d(0) ):
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
        Y_in = np.array(y_in)
        Y_in = Y_in.reshape(Y_in.size, 1)
        if (y_in == 0).all():
            Y_in = np.zeros([self.n_out,1])
        if Input.size == self.n_in:                
            z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
            if self.output_feedback:
                z += np.dot(self.Wor,Y_in)
            if self.noise > 0:
                z += np.random.normal(0, self.noise, [self.neu, 1])                
            self.a = (1-self.leakrate)*self.a + self.leakrate*np.tanh(z)
            
            a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
            y = np.dot(self.Wro,a_wbias)
            return y
        else:
            raise ValueError("input must have size n_in")

    def copy_weights(self, Rede):
        if self.Wro.shape == Rede.Wro.shape:
            self.Wro = np.copy(Rede.Wro)
            self.Wrr = np.copy(Rede.Wrr)
            self.Wir = np.copy(Rede.Wir)
            self.Wbr = np.copy(Rede.Wbr)
        else:
            print("shapes of the weights are not equal")

    def save_reservoir(self,fileName):
            data = {}
            data['Wrr'] = self.Wrr
            data['Wir'] = self.Wir
            data['Wbr'] = self.Wbr
            data['Wro'] = self.Wro
            data['a0'] = self.a
            io.savemat(fileName,data)

    def load_reservoir(self,fileName):
        data = {}
        io.loadmat(fileName, data)
        self.load_reservoir_from_array(data)

    def load_reservoir_from_array(self, data):        
        self.Wrr = data['Wrr']
        self.Wir = data['Wir']
        self.Wbr = data['Wbr']
        self.Wro = data['Wro']
        if 'a0' in data:  # check by Eric
            self.a = data['a0']
            
        # added by Eric - start
        if 'Wro_b' in data:
            self.Wro = np.hstack((data['Wro_b'], self.Wro))

        if 'leak_rate' in data:
            try:
                self.leakrate = data['leak_rate'][0][0]
            except:
                self.leakrate = data['leak_rate']
        self.neu = self.Wrr.shape[0]
        self.n_in = self.Wir.shape[1]
        self.n_out = self.Wro.shape[0]            
        # added by Eric - end
        

    def load_initial(self,filename):
        data = {}
        print("loading reservoir")
        io.loadmat(filename, data)
        self.Wrr0 = data['Wrr']
        self.Wir0 = data['Wir']
        self.Wbr0 = data['Wbr']
        self.Wro = data['Wro']
        #self.Wor0 = data['Wor']
        self.a = data['a0']

    def save_initial_fun(self,filename):
        data = {}
        data['Wrr'] = self.Wrr0
        data['Wir'] = self.Wir0
        data['Wbr'] = self.Wbr0
        data['Wor'] = self.Wor0
        data['Wro'] = self.Wro
        data['a0'] = self.a
        print( "saving reservoir")
        io.savemat(filename, data)

    
    def trainLMS(self,ref):

        learningrate = 1

        e = self.trainingError(ref)
        for saida in range(self.n_out):
            Theta = self.Wro[saida, :]
            Theta = Theta.reshape([self.neu, 1])
            Theta = Theta - learningrate*e*self.a/np.dot(self.a.T,self.a)
            self.Wro[saida, :] = Theta.T


    def offline_training(self,X,Y,regularization,warmupdrop): #X is a matrix in which X[i,:] is all parameters at time i. Y is a vector of desired outputs.
        A = np.empty([Y.shape[0]-warmupdrop,self.neu])
        for i in range(Y.shape[0]):
            if self.output_feedback:
                if i > 0:
                    self.update(X[i,:],Y[i-1,:])
                else:
                    self.update(X[i,:])
                if i > warmupdrop:
                    A[i-warmupdrop,:] = self.a.T
            else:
                self.update(X[i, :])
                if i > warmupdrop:
                    A[i - warmupdrop, :] = self.a.T
                    
                    
        A_wbias = np.hstack((np.ones([A.shape[0],1]),A))

        P = np.dot(A_wbias.T,A_wbias)
        R = np.dot(A_wbias.T,Y[warmupdrop:])
            #print R,"R"
        Theta = np.linalg.solve(P+regularization*np.eye(self.neu+1,self.neu+1),R)
        self.Wro = Theta.T

    def reset(self):

        self.a = np.zeros([self.neu, 1])

    def get_forgetingfactor(self):

        return self.forget
    
    def get_state_from(rede):
        
        self.a = rede.a
        
    def get_derivative_df_du(self, inp,y_in = np.atleast_2d(0)):
        
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)
        #Y_in = np.zeros([self.n_out,1])
        z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
        if not (y_in == 0).all():
            Y_in = np.array(y_in)
            Y_in = Y_in.reshape(Y_in.size, 1)
            z += np.dot(self.Wor,Y_in)
                     
        J = grad_tanh(z)
        
        return np.dot(J,self.Wir)
    
    def get_derivative_df_dx(self, inp,y_in = np.atleast_2d(0)):
        
        Input = np.array(inp)
        Input = Input.reshape(Input.size,1)

        z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr
        if not (y_in == 0).all():
            Y_in = np.array(y_in)
            Y_in = Y_in.reshape(Y_in.size, 1)
            z += np.dot(self.Wor,Y_in)
        
        J = grad_tanh(z)

        z1 = self.Wrr
        if self.output_feedback:
            z1 += np.dot(self.Wor,self.Wro[:,1:])
        z1 = np.dot(J,z1)
        
        return (1-self.leakrate)*np.eye(self.neu) + self.leakrate * z1
        
    def get_current_output(self):
        a_wbias = np.vstack((np.atleast_2d(1.0),self.a))
        return np.dot(self.Wro,a_wbias)
    
    def covariance_reset(self,diag_alpha):
        self.P = np.eye(self.neu)/diag_alpha
    

    @staticmethod
    def new_rnn_from_weights(weights):
        esn_4tanks = EchoStateNetwork(
            neu = 400,
            n_in = 2,
            n_out = 2,
            gama = 0.1,
            ro = 0.99,
            psi = 0.0,
            in_scale = 0.1,
            bias_scale = 0.1,
            initial_filename = "4tanks1",
            load_initial = False,
            save_initial = False,
            output_feedback = False)
        esn_4tanks.load_reservoir_from_array(weights)
        esn_4tanks.reset()
        return esn_4tanks
