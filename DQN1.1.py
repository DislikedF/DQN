import subprocess
import time
import io as IO
from skimage import data, color, transform, io
from skimage.viewer import ImageViewer
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np           
import random                
import socket
import tensorflow as tf      


from collections import deque
import matplotlib.pyplot as plt


### TRAINING HYPERPARAMETERS
total_episodes = 2800        # Total episodes for training
max_steps = 10000              # Max possible steps in an episode
batch_size = 64
action_size = 5              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

training = True

stack_size = 4 #stack of 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

#Set up server side
s = socket.socket()         # Create a socket object
host = '0.0.0.0'            # Get local machine name
port = 12345                # Reserve a port
print ('Server started!')
print ('Waiting for clients...')

s.bind((host, port))        # Bind to the port
s.listen(5)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]
    
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None,*state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [64] , name="actions_")
            
            # target_Q R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
           
            #First convnet:
        
            # Input is 84x84x4
            self.conv1 = tf.keras.layers.Conv2D(
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")(self.inputs_)
           
           
            self.conv1_batchnorm = tf.keras.layers.BatchNormalization(               
                                                   epsilon = 1e-5,
                                                   trainable = True,
                                                   name = 'batch_norm1')(self.conv1)
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            
            
           
            #Second convnet:
           
            self.conv2 = tf.keras.layers.Conv2D(
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")(self.conv1_out)
            
            self.conv2_batchnorm = tf.keras.layers.BatchNormalization(
                                                   trainable = True,
                                                   epsilon = 1e-5,
                                                   name = 'batch_norm2')(self.conv2)

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            
            
           
            #Third convnet:
           
            self.conv3 = tf.keras.layers.Conv2D(
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")(self.conv2_out)
            
            self.conv3_batchnorm = tf.keras.layers.BatchNormalization(
                                                   trainable = True,
                                                   epsilon = 1e-5,
                                                   name = 'batch_norm3')(self.conv3)

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            
            
            
            self.flatten = tf.keras.layers.Flatten()(self.conv3_out)
            
            
            
            self.fc = tf.keras.layers.Dense(
                                  units = 512,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")(self.flatten)
            
            
            self.output = tf.keras.layers.Dense( 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = 1, 
                                           activation=None)(self.fc)

  
            #predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is difference between predicted Q_values and Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

          #Send command to save screen, then read saved file          
def      readimage(c):
      try:
            sendmessage("S", c)
            time.sleep(0.1)
            image = io.imread("C:/Users/ryanf/Documents/ProjectAITestLevelDQN/AITestLevel_Data/SomeLevel.PNG", as_gray=True)
            image = transform.resize(image, [84,84])
            
            return image
      except:
            print("image error")

        #Recieve message from test game application 
def     recievemessage(c):
     try:
            rmsg = c.recv(20).decode()
            intmsg = int(rmsg)
            print(intmsg)
            return intmsg
     except:
        pass

        #Send message to test game application
def     sendmessage(smsg, c):
     try:
            c.send(str(smsg).encode())
            time.sleep(0.05)
     except socket.timeout:
          pass

    #Stack preprocesed frames
def stack_frames(stacked_frames, state, is_new_episode):
    preprocessed_frame = state
    frame = preprocessed_frame
    
    
    if is_new_episode:
        # Clear stacked frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        #new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque
        stacked_frames.append(frame)

        # stacked state
        stacked_state = np.stack(stacked_frames, axis=2)
        
    
    return stacked_state, stacked_frames



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ##randomize a number
    exp_exp_tradeoff = np.random.rand()

    #epsilon greedy strategy
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability

#open game environment
def opengame():
    env = subprocess.Popen("C:/Users/ryanf/Documents/ProjectAITestLevelDQN/AITestLevel.exe", stdout=subprocess.PIPE)
    time.sleep(5)
   #possible actions
    left = 1
    right = 2
    shoot = 3
    up = 4
    down = 5
   
    possible_actions = ([left, right, shoot, up, down])
    c, addr = s.accept()     # Establish connection with client.
    c.setblocking(1)
    c.settimeout(100)
    print ('Got connection from', addr)
    return c, possible_actions

c, possible_actions = opengame()
c.settimeout(2.5)

# Reset graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = readimage(c)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    
    # Random action
    action = (random.choice(possible_actions))
    
    # Get reward
    sendmessage(action, c)
    print (action)
    sendmessage("R", c)
    reward =  recievemessage(c)
    
    # is episode finsihed?
    if reward >= 999:
        done =  True
        reward = 0;
    elif reward != 999:
            done = False
        
    
    # If done
    if done == True:
        print("DONE TRIGGERED")
        
        next_state = np.zeros(state.shape)
        
        #Add experience to memory
        memory.add((state, action, reward, next_state, done))
             
        #state
        state = readimage(c)
        state = state
        
        #Stackframes
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        
    else:      
        next_state = readimage(c)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        
        # state now  next state
        state = next_state
        
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

#Loss
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# Saver to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (use to reduce epsilon) 
        decay_step = 0

        for episode in range(total_episodes):
            # Set step 0
            step = 0
            
            # Initialize rewards
            episode_rewards = []
            
            # Make a new episode and observe first state
            state = readimage(c) 
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            print("Not every time")
            
            while step < max_steps:
                step += 1
                
                # Increase decay_step
                decay_step +=1
                
                # Predict the action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Do action
                sendmessage(action, c)
               
                sendmessage("R", c)
                reward = recievemessage(c)
                

                # Look if the episode is finished
                if reward >= 999:
                    done = True
                    reward = 0;
                elif reward != 999:
                    done = False
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done == True:
                    # the episode ends so no next state
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))
                    done = False

                else:
                    # Get the next state
                    next_state = readimage(c)
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    
                    # st+1 is now our current state
                    state = next_state


                # LEARNING            
                # Obtain random mini-batch from memory
               
                batch = memory.sample(batch_size)
                
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch], ndmin=1)
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                
                target_Qs_batch = []
                
                
                 # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If in terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])
               
               
                
                
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={ DQNetwork.inputs_: states_mb,
                                                DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb})
                                 

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


    
