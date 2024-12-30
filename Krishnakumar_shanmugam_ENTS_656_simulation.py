import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import time

'''#Base Station Co-ordinates'''
alpha_coordinate = [0,1]
beta_coordinate = [np.sqrt(3)/2,-1/2]

'''#Inputs for computing EIRP'''
height_BTS = 50 #in meters
TX_power = 47 #in dBm
L_loss = 1 #in db
G_TX = 14.8 #in dbi (Boresight)
frequency = 1910 #in MHZ

#Parameters at Mobile RX
height_RX = 1.5 #in meters
HO_margin = 3 #dB 
RSLthresh = -102 #dBm
EIRP_src = TX_power + G_TX - L_loss  #in dBm
    
time_index = 0;
time_duration = 10
user_velocity = 15
north = 0
south = 0

'''Define the range and number of users'''
lower_limit = -1500
upper_limit = 1500
num_users = 640
interval = 100
time_duration  = 3600*4

'''reading antennae info from file'''
antenna_pattern = np.loadtxt('./antenna_pattern.txt')
road_start = lower_limit
road_end  = upper_limit
rsl_serv=0
rsl_neigh=0

'''Inputs for the x , y co-ordinates'''
x_value = 20       # Fixed x-coordinate
y_min = -lower_limit # Minimum y-coordinate
y_max = upper_limit # Maximum y-coordinate
y_step = 1         # Step size for the y-coordinate
     
''' parameters of User Call info '''
user_call_rate = 2 #2 calls per hall
avg_call_duration = 180 #3mins per call
prb_of_call_set_up = user_call_rate / (60*60)

''' Enum list for the sector Info '''
class serving_sector_enum(Enum):
    SERV_SECTOR  = 1
    NEIGHBOR_SECTOR = 2  
class sector_enum(Enum):
    ALPHA_SECTOR = 0
    BETA_SECTOR = 1
''' Enum list for the List operations '''    
class param_opn(Enum):
    INCREMENT = 0
    DECREMENT = 1
    APPEND    = 2
    DELETE    = 3
    RESET     = 4     
    
''' Active call user list for each sectors '''
alpha_user_list =[]
beta_user_list  =[]
'''# archive_list stores info = (user_index, initial user coordinate , call_duration of user )'''
alpha_archive_list = []       
beta_archive_list = [] 

''' USER_METRIC_A: user_info[0]: Y coordinate, 
    USER_METRIC_B: user_info[1]: direction of movement , 
    USER_METRIC_C: user_info[2]: current position 
    USER_METRIC_D: user_info[3]: remaining timer duration ,  
    USER_METRIC_E: user_info[4]: active call in progress
    USER_METRIC_F: user_info[5]: serving sector ID
    USER_METRIC_G: user_info[6]: call time stamp
    USER_METRIC_H: user_info[7]: movement tracking 
'''
''' User info Numpy array used for functional processing  '''
user_info = np.zeros((num_users,8))
user_info[:,0]= np.random.uniform(lower_limit, upper_limit + 1, size=num_users) #Generate
user_info[:,1]= np.random.randint(0,2, size=num_users)
user_info[:,2] = user_info[:,0] 

'''
    #[SECTOR_METRIC_A]0: ++ No. of channels currently in use
    #[SECTOR_METRIC_B]1: ++ number of call attempts
    #[SECTOR_METRIC_C]2: ++ No. of successful calls
    #[SECTOR_METRIC_D]3: ++ Number of successful HO Attempt 
    #[SECTOR_METRIC_E]4: No. of HO failuresß into and out of each sector
    #[SECTOR_METRIC_F]5: record the call drop due to low signal strength 
    #[SECTOR_METRIC_G]6: record the number of blocks due to capacity
    #[SECTOR_METRIC_H]7: record the number of HO attempts
'''
''' Initialise the Sector numpy array info '''
sector_info = np.zeros((2,8))
    

''' Sector Metric info used for tabulation of Trace Info'''
metrics = [
    "No. of channels currently in use",
    "No. of call attempts",
    "No. of successful calls",
    "No. of successful HO",
    "No. of HO failures into and out of each sector",
    "Call drops due to low signal strength",
    "Blocks due to capacity",
    "No. of HO attempts"
]    

''' Function to update the Sector Info parameters '''
def update_sector_metric(sector_metric_id: int, current_serving_sector: int , parameter_sector: int , type_set: int):
    match (sector_metric_id):
        case 0 | 1 | 2: #0 sector_metric_enum.SECTOR_METRIC_A ++ No. of channels currently in use; #1 sector_metric_enum.SECTOR_METRIC_B: #++ number of call attempts; "2" #sector_metric_enum.SECTOR_METRIC_B: #++ number of call attempts
            if(type_set  == 0):
                sector_info[parameter_sector][sector_metric_id] +=1 
            elif(type_set  == 1):
                if(sector_info[parameter_sector][sector_metric_id]!=0):
                    sector_info[parameter_sector][sector_metric_id]-=1
            elif(type_set  == 4):
                sector_info[parameter_sector][sector_metric_id] = 0
                           
        case 3 | 4 | 5: #(3)sector_metric_enum.SECTOR_METRIC_D: #++ Number of successful HO Attempt ;(4)#sector_metric_enum.SECTOR_METRIC_E: #No. of HO failures into and out of each sector ; #(5)sector_metric_enum.SECTOR_METRIC_F: #record the call drop due to low signal strength and capacity
            if (current_serving_sector == sector_enum.ALPHA_SECTOR):
                    parameter_sector = sector_enum.BETA_SECTOR
            elif(current_serving_sector == sector_enum.BETA_SECTOR):
                    parameter_sector = sector_enum.ALPHA_SECTOR
                    
            if(type_set  == 0):
                sector_info[parameter_sector,sector_metric_id]+=1 
            elif(type_set  == 2):
                sector_info[parameter_sector,sector_metric_id] = 0
        
        case 6 | 7:#(6) sector_metric_enum.SECTOR_METRIC_G: #record the number of blocks due to capacity; # (7)sector_metric_enum.SECTOR_METRIC_H: #mark call attempt failure on serving sector due to low signal strength
            sector_info[parameter_sector,sector_metric_id]+=1 
                 
''' Function to update the User list (1.Append to active list 2. Delete from Active list / move to archive list) '''
def update_user_list(type_set: int, user_idx: int):
    # Fetch the serving sector ID
    serving_sector_num = int(user_info[user_idx][5])
    
    '''Determine the list and archive based on the serving sector'''
    if serving_sector_num == 0:
        sector_list = alpha_user_list
        sector_archive = alpha_archive_list
    else:
        sector_list = beta_user_list
        sector_archive = beta_archive_list

    '''Common user data'''
    initial_user_coordinate = user_info[user_idx][0]
    call_duration = user_info[user_idx][3]
    call_time_stamp = user_info[user_idx][6]
    
    if type_set == param_opn.APPEND:
        # Mark this user under active call list
        if user_idx not in sector_list:
            sector_list.append(user_idx)
    elif type_set == param_opn.DELETE:
        # Delete user from active list and archive
        if user_idx in sector_list:
            sector_list.remove(user_idx)
            sector_archive.append((user_idx, initial_user_coordinate, call_duration, call_time_stamp))

''' Throuhput computation category list '''
alpha_si_green_list =[]
alpha_si_magenta_list =[]
alpha_si_red_list =[]
beta_si_green_list =[]
beta_si_magenta_list =[]
beta_si_red_list =[]

''' Part1 Simulation Array '''
result=[]
eirp_net_d = []
eirp_net_d2 = []
eirp_net_value=[]
eirp_net_value2=[]
eirp_net_cost231_val1=[]
eirp_net_cost231_val2=[]
only_cost231_1=[]
only_cost231_2=[]
only_cost231_d1 =[]
only_cost231_d2 =[]
shadow_rank_array =[]
eirp_shadowing_val1 =[]
eirp_shadowing_val2 =[]
eirp_fading_val1 =[]
eirp_fading_val2 =[]
theta_list = [] 

''' Function to update SI value list for every position of User coordinate on the road '''
def generate_si_list(si_value_db,user_idx, serving_sector):
    if serving_sector == 0: #alpha sector
        if(si_value_db > 10):
            alpha_si_green_list.append(user_info[user_idx][2])
        elif(si_value_db > 5):
            alpha_si_magenta_list.append((user_info[user_idx][2]))
        else:
            alpha_si_red_list.append(user_info[user_idx][2])
    else:
        if(si_value_db > 10):
            beta_si_green_list.append(user_info[user_idx][2])
        elif(si_value_db > 5):
            beta_si_magenta_list.append(user_info[user_idx][2])
        else:
            beta_si_red_list.append(user_info[user_idx][2])

''' Function to generate the User coordinate list based upon the user input '''
def generate_coordinates(x_value, y_min, y_max, y_step):
    # Constants
    x_value = 20       # Fixed x-coordinate
    y_min = -1500   # Starting y-coordinate
    y_max = 1500      # Ending y-coordinate
    y_step = 1         # Step size for the y-coordinate
    y_values = np.arange(y_min, y_max + 1, y_step)  # Generate the y-coordinates from -1500 to 1500, inclusive
    coordinates = np.array([[x_value, y] for y in y_values]) #(list of coordinate pairs)
    return coordinates

''' Function to compute the distance '''
def calculate_distance(x1,y1,x2,y2):
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return(d)

''' Function to search specific value from array 
Argument  Value : '''
def find_value_array(value, array):
    # Convert to a numpy array if not already
    array = np.array(array)
    # Search for the value and return the second column
    result = np.where(array == value)
    if result[0].size > 0:
        return array[result[0][0], 1]
    else:
        return 0

''' Function to generate rank of Repeated range of shadow values at every user coordinate position '''
def location_shadow_rank(mean,sigma,shadow_size,length_road):
    '''Generate the y-values from -lower limit to upper limit (inclusive)'''
    y_values = np.arange(-length_road/2, (length_road/2)+1,1)
    '''Generate or provide the rank array with 300 values'''
    rank_array = np.random.normal(mean,sigma,shadow_size)  
    '''# Calculate rank if it is repeated evry 10m 
    we have 300 values and 3001 y-values, the rank array will be repeated uniformly'''
    repeat_interval = len(y_values) // len(rank_array)
    '''Assign ranks to the y-values'''
    rank_idx = 0
    for i in range(len(y_values)):
        '''For each y-value, append the corresponding rank from rank_array'''
        result.append((y_values[i], rank_array[rank_idx]))
        ''' After `repeat_interval` steps, move to the next rank'''
        if (i + 1) % repeat_interval == 0 and rank_idx < len(rank_array) -1:
            rank_idx += 1
    result_array = np.array(result)
    return(result_array)
    
''' Function to estimate the PL using Cost231 at the RSL '''
def propagation_loss_cost231(d):
    PL50 = 0
    if frequency > 1500:
        a_hm = (1.1*np.log10(frequency)-0.7)*height_RX - (1.56*np.log10(frequency)-0.8)
        PL50 = 46.3 + 33.9*np.log10(frequency) - 13.82*np.log10(height_BTS) + (44.9-6.55*np.log10(height_BTS))*np.log10(d/1000)-a_hm
    else:
        print("PCS band < 1500 Hz. Not matching for COST-231 computation:  ")
    return(PL50)    

''' Function to estimate the Fading value instantly '''
def fading_generation(mean,sigma,num_samples):
    '''Generate Rayleigh samples '''
    real_part = np.random.rayleigh(1, num_samples)
    sorted_raleigh_samples = np.sort(real_part)
    return(20*np.log10(sorted_raleigh_samples[1]))


def EIRP_net(A,B):
    dot_product = np.dot(A,B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    value = dot_product / ((magnitude_A)*(magnitude_B))
    radians_value = (np.arccos(value))    
    '''Convert radians to degrees manually'''
    theta_val = round(np.degrees(radians_value))
    '''Search for the value and return the second column'''
    result = np.where(antenna_pattern[:,0] == theta_val)
    if result[0].size > 0:
        antennae_disc = antenna_pattern[result[0][0], 1]
    else:
        antennae_disc = 0
    theta_list.append((A,B[1],theta_val,antennae_disc))    
    EIRP_net_value = EIRP_src - antennae_disc
    return(EIRP_net_value)

def generate_eirp_net_graph(BTS_coordinate,BETA_coordinate):
    print("generate_eirp_net_graph")
    
    for i in range(len(coordinates)):    
        eirp_net_d.append(calculate_distance(BTS_coordinate[0],BTS_coordinate[1], coordinates[i,0], coordinates[i,1]+1500))    
        eirp_net_value.append(EIRP_net(BTS_coordinate,coordinates[i]))
    
    for i in range(len(coordinates)):    
        eirp_net_d2.append(calculate_distance(BETA_coordinate[0],BETA_coordinate[1], coordinates[i,0], coordinates[i,1]+1500))    
        eirp_net_value2.append(EIRP_net(BETA_coordinate,coordinates[i]))
    
    plt.figure()
    plt.plot(eirp_net_d, eirp_net_value, color='b', label="Alpha sector")
    plt.plot(eirp_net_d2, eirp_net_value2, color='r', label="Beta sector")

    # Labeling the axes
    plt.xlabel('Distance (meters)')
    plt.ylabel('EIRP (dB or units)')
    plt.title('Distance vs EIRP Graph')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def generate_eirp_cost231():
    print("generate_eirp_cost231")
    rsl = 0
    for i in range(len(coordinates)):      
        distance = (calculate_distance(alpha_coordinate[0],alpha_coordinate[1], coordinates[i,0], coordinates[i,1]))  
        only_cost231_d1.append(distance)
        only_cost231_1.append(propagation_loss_cost231(distance))
        rsl = eirp_net_value[i] - propagation_loss_cost231(distance)
        eirp_net_cost231_val1.append(rsl)
    
    for i in range(len(coordinates)):        
        distance = (calculate_distance(beta_coordinate[0],beta_coordinate[1], coordinates[i,0], coordinates[i,1]))  
        only_cost231_d2.append(distance)
        only_cost231_2.append(propagation_loss_cost231(distance))
        rsl = eirp_net_value2[i] - propagation_loss_cost231(distance)
        eirp_net_cost231_val2.append(rsl)
    
    plt.figure()
    plt.plot(eirp_net_d, eirp_net_cost231_val1,  color='b', label="Alpha sector")
    plt.plot(eirp_net_d2, eirp_net_cost231_val2, color='r', label="Beta sector")
    
    # Labeling the axes
    plt.xlabel('Distance (meters)')
    plt.ylabel('EIRP (dB)')
    plt.title('Distance vs EIRP_COST231')
    plt.grid(True)
    plt.legend()
    plt.show()

def generate_eirp_shadowing():
    print("generate_eirp_shadowing")
    rsl = 0
    for i in range(len(coordinates)):     
        shadow_loss = find_value_array(coordinates[i,1],shadow_rank_array)
        rsl = eirp_net_cost231_val1[i] - shadow_loss
        eirp_shadowing_val1.append(rsl)
    
    for i in range(len(coordinates)):     
        shadow_loss = find_value_array(coordinates[i,1], shadow_rank_array)
        rsl = eirp_net_cost231_val2[i] - shadow_loss
        eirp_shadowing_val2.append(rsl)
    
    plt.figure()
    plt.plot(eirp_net_d, eirp_shadowing_val1,  color='b', label="Alpha sector")
    plt.plot(eirp_net_d2, eirp_shadowing_val2, color='r', label="Beta sector")
    plt.xlabel('Distance (meters)')
    plt.ylabel('EIRP (dB')
    plt.title('Distance vs SHADOW')
    plt.grid(True)
    plt.legend()
    plt.show()

def generate_eirp_fading():
    print("generate_eirp_fading")
    rsl = 0
    for i in range(len(coordinates)):     
        fade_value = fading_generation(0,1,10)
        rsl = eirp_shadowing_val1[i] + fade_value
        eirp_fading_val1.append(rsl)
        
    for i in range(len(coordinates)):    
        fade_value = fading_generation(0,1,10)
        rsl = eirp_net_cost231_val2[i] + fade_value
        eirp_fading_val2.append(rsl)
    plt.figure()
    plt.plot(eirp_net_d, eirp_fading_val1,  color='b', label="alpha sector")
    plt.plot(eirp_net_d2, eirp_fading_val2, color='r', label="beta sector")
    plt.xlabel('Distance (meters)')
    plt.ylabel('EIRP (dB)')
    plt.title('Distance vs FADE')
    plt.grid(True)
    plt.legend()
    plt.show()

''' PART1 SIMULATION '''
print("PART1 SIMULATION in progress")
coordinates = generate_coordinates(x_value, y_min, y_max, y_step)
generate_eirp_net_graph(alpha_coordinate,beta_coordinate)
generate_eirp_cost231()
shadow_rank_array = location_shadow_rank(0,2,300,3000)
generate_eirp_shadowing()
generate_eirp_fading()
print("PART1 SIMULATION Completed ")
print("PART2 SIMULATION STARTS")
test_len = int(input("Enter the Road length (in km) to be run for simulation : Enter 3 or 6    "))
num_users = int(input("Enter the user traffic for simulation : 160 or 640 "))
if test_len == 3:
     lower_limit = -1500
     upper_limit = 1500
elif test_len == 6:
     lower_limit = -3000
     upper_limit = 3000

''' PART2 SIMULATION for executing multiple users in mobility '''
def display_sector_table():
    print("*-" * 45)
    print(f"{'Parameter':<60} {'Alpha Sector':<15} {'Beta Sector':<15}")
    print("*-" * 45)
    for i, metric in enumerate(metrics):
        alpha_value = sector_info[0, i]
        beta_value = sector_info[1, i]
        print(f"{metric:<60} {alpha_value:<15} {beta_value:<15}")
    print("*-" * 45)

print("PART2 SIMULATION in progress")
start_time = time.time()
''' PART2 Main Simulation '''
while(time_index < time_duration):   
    user_index = 0
    '''Display the sector table info at the end of every hour '''
    if((time_index > 0) and (time_index % 3600 == 0)):
        print("\n\n\nvalue of sector metrics at the end of", (time_index/3600),"Hr")
        display_sector_table()
    while (user_index < num_users):    
        ''' Path 1  (call is not yet established) / [USER_METRIC_F]:Call is not setup for the user'''
        if(user_info[user_index][4]== 0 ):
            '''Step 1 :  Generate if the user has prob to make call '''
            user_prob_gen3 = np.random.uniform(0, 1)
            user_prob_gen3_ref = 1/time_duration
            
            if(user_prob_gen3 < prb_of_call_set_up):
                user_coordinate = (x_value,user_info[user_index][2])

                eirp_net_alpha = EIRP_net(alpha_coordinate,user_coordinate)
                eirp_net_beta = EIRP_net( beta_coordinate,user_coordinate)
                
                distance_serv = (calculate_distance(alpha_coordinate[0],alpha_coordinate[1], x_value, user_info[user_index][2]))          
                distance_neigh = (calculate_distance(beta_coordinate[0],beta_coordinate[1], x_value, user_info[user_index][2]))          
            
                rsl_alpha = eirp_net_alpha - propagation_loss_cost231(distance_serv)
                rsl_beta = eirp_net_beta - propagation_loss_cost231(distance_neigh)

                '''Step 1 :  Generate RSL for serving, neighbor. decide the RSL strongest sector  (alpha or beta)'''
                if(rsl_alpha > rsl_beta): 
                    rsl_serv = rsl_alpha  #'''storing the RSL value'''
                    rsl_neigh = rsl_beta
                    user_info[user_index][5] = 0 #'''update the serving sector ID as alpha'''
                else:
                    '''storing the RSL value'''
                    rsl_serv = rsl_beta
                    rsl_neigh = rsl_alpha
                    '''update the serving sector ID as beta '''
                    user_info[user_index][5] = 1 
                si_value  = rsl_serv - rsl_neigh #compute SI_value
                
                '''Step 2 :  Check RSL serving > RSLthreshold: '''
                if(rsl_serv > RSLthresh):
                    '''#  if yes :consider call attempt progress [SECTOR_METRIC_B]: ++ number of call attempts on serving sector '''
                    update_sector_metric(1,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                    serving_sector_num  = int(user_info[user_index][5])
                    ''' check the Availability of number of channels in serving '''
                    if(sector_info[serving_sector_num][0]<15):
                        '''#Call is allowed. Update [SECTOR_METRIC_A]: ++ No. of channels currently in use'''
                        update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                        ''' Update active user call list '''
                        update_user_list(param_opn.APPEND,user_index)
                        ''' USER_METRIC_E: call is set up for the user '''
                        user_info[user_index][4] = 1
                        '''# [USER_METRIC_D]: calculate the Remaining timer duration'''
                        user_info[user_index][3] = np.random.randint(0,(avg_call_duration+1)) + time_index
                        '''# [USER_METRIC_G]:  Update the user call time stamp of call start '''
                        user_info[user_index][6] = time_index
                    else:
                        '''#[SECTOR_METRIC_G]: record number of blocks due to capacity (serving sector)'''
                        update_sector_metric(6,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                ''' Call is not set up from identified Serving sector. Start exploring Neighbor sector '''
                if(user_info[user_index][4]== 0):
                #else:
                    '''#  else: RSL below RSLthreshold '''
                    '''#call attempt failure on this user and move to another user
                    # [SECTOR_METRIC_F]: mark call attempt failure on serving sector due to low signal strenght'''
                    update_sector_metric(5,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                    ''' identify the neighbor sector '''
                    neigh_sector_num = 0 if user_info[user_index][5] == 1 else 1
                    
                    '''#Step 4 :  Check if call can be progressed on neighbor sector > RSLthreshold'''
                    if(rsl_neigh > RSLthresh):
                        '''#[SECTOR_METRIC_B]: ++ number of call attempts'''
                        update_sector_metric(1,int(user_info[user_index][5]),neigh_sector_num,0)
                        '''# if we have available channel on neighbor sector'''
                        if(sector_info[neigh_sector_num][0]<15):
                              '''# [SECTOR_METRIC_A]:  ++ No. of channels currently in use'''
                              update_sector_metric(0,int(user_info[user_index][5]),neigh_sector_num,0)
                              '''# [USER_METRIC_C]: calculate the Remaining timer duration'''
                              user_info[user_index][3] = np.random.randint(0,181) + time_index
                              '''# [USER_METRIC_F]:Call is setup for the user'''
                              user_info[user_index][4] = 1
                              ''' [USER_METRIC_E] update the serving sector ID for the user '''
                              user_info[user_index][5] = neigh_sector_num
                              '''# [USER_METRIC_G]:  Update the user call time stamp '''
                              user_info[user_index][6] = time_index
                              ''' #Step 2:   Calculate the direction and the next position'''
                        else:
                            '''#else No:  [SECTOR_METRIC_G]: record number of blocks due to capacity '''
                            update_sector_metric(6,int(user_info[user_index][5]),neigh_sector_num,0)
                    else:
                        '''#else [SECTOR_METRIC_F]:  No: RSL Neighbor < RSL threshold #[SECTOR_METRIC_H]: mark call attempt failure on serving sector due to low signal strength'''
                        update_sector_metric(5,neigh_sector_num,neigh_sector_num,0)
            else:   
              pass #Step 2 : else (user has no prob. to make the call ; move to next position and next user) 
        elif(user_info[user_index][4]== 1 ):
            '''#Path 2 (active ongoing call ) / [USER_METRIC_F]:Call is setup for the user'''
            #if remaining timer duration is expired: 
            if(user_info[user_index][3] == time_index):
                '''# [SECTOR_METRIC_A]:  -- No. of channels currently in use'''
                update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),1)
                ''' # [SECTOR_METRIC_C]: ++ No. of successful calls(serving sector)'''
                update_sector_metric(2,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                '''#move the call to archived list'''
                update_user_list(param_opn.DELETE,user_index)
                '''[USER_METRIC_E]:Call is reset for call setup of the user, [USER_METRIC_F:] reset of the serving sector ID for the user,# [USER_METRIC_G]:  reset Update the user call time stamp '''
                user_info[user_index][3:7]= 0 
                #STEP 2: Move to direction and consider to generate the new co-ordinate position
            else:            
                '''#Path 3: if the expiry call duration has not reached and identify the serving sector / [USER_METRIC_F]:Call is not setup for the user && [USER_METRIC_C]:  timer expired'''
                serving_sector_num  = int(user_info[user_index][5])
                user_coordinate = (x_value,user_info[user_index][2])
                eirp_net_alpha = EIRP_net(alpha_coordinate,user_coordinate)
                eirp_net_beta = EIRP_net( beta_coordinate,user_coordinate)
                
                distance_serv = (calculate_distance(alpha_coordinate[0],alpha_coordinate[1], x_value, user_info[user_index][2]))          
                distance_neigh = (calculate_distance(beta_coordinate[0],beta_coordinate[1], x_value, user_info[user_index][2]))          
                fade_value_alpha = fading_generation(0,1,10)
                fade_value_beta = fading_generation(0,1,10)

                '''Search for the shadow value in relation to y-coordinate'''
                result = np.where(shadow_rank_array == user_info[user_index][2])
                if result[0].size > 0:
                    shadow_loss = shadow_rank_array[result[0][0], 1]
                else:
                    shadow_loss = 0
               
                rsl_alpha = eirp_net_alpha - propagation_loss_cost231(distance_serv) + fade_value_alpha + shadow_loss
                rsl_beta = eirp_net_beta - propagation_loss_cost231(distance_neigh) + fade_value_beta + shadow_loss
                ''' update the serving and neighbor info''' 
                rsl_serv = rsl_alpha if serving_sector_num == 0 else rsl_beta 
                rsl_neigh = rsl_beta if serving_sector_num == 0 else rsl_alpha
              
                '''#check If RSL serving  > RSLneighbour'''
                if((rsl_serv > rsl_neigh) and (rsl_serv > RSLthresh)):
                    pass #rsl_serv > RSLthresh: Continue call ") # move to the direction part and the next position part 
                else: #else check RSL serving < RSL neighbour check HO conditions ")
                    ''' identify the neighbor sector '''
                    neigh_sector_num = 0 if user_info[user_index][5] == 1 else 1
                   
                    '''#1.2 If check for RSLneighbor > RSLserving + HO_m'''
                    if(rsl_neigh > (rsl_serv + HO_margin)):
                        '''# [SECTOR_METRIC_H]: ++ Number of HO Attempt (serving sector)'''
                        update_sector_metric(7,int(user_info[user_index][5]),serving_sector_num,0)
                        '''1.2.1 Check for channel availability in neighbor sector'''
                        if(sector_info[neigh_sector_num][0]<15):
                            '''# [SECTOR_METRIC_A]:  ++ No. of channels currently in use (serving sector)'''
                            update_sector_metric(0,int(user_info[user_index][5]),serving_sector_num,1)
                            '''# [SECTOR_METRIC_A]:  -- No. of channels currently in use (neighbor sector)'''
                            update_sector_metric(0,int(user_info[user_index][5]),neigh_sector_num,0)
                            '''# [SECTOR_METRIC_D]: ++ Number of successful HO (serving sector)'''
                            update_sector_metric(3,int(user_info[user_index][5]),serving_sector_num,0)
                            '''#move the call to archived list'''
                            update_user_list(param_opn.DELETE,user_index)
                            '''[USER_METRIC_E]: Update the serving sector ID from Serving to neighbor sector'''
                            user_info[user_index][5] = neigh_sector_num
                            '''[SECTOR_METRIC_I]: ++ add this user to active call list(Neighbor sector)'''
                            update_user_list(param_opn.APPEND,user_index)
                        else:
                            '''HO : No Channels available in Neighbor sector '''
                            '''#[SECTOR_METRIC_E]: No. of HO failures into and out of each sector'''
                            update_sector_metric(4,int(user_info[user_index][5]),serving_sector_num,0)
                            '''#[SECTOR_METRIC_G]: record the number of blocks due to capacity '''
                            update_sector_metric(6,int(user_info[user_index][5]),neigh_sector_num,0)
                            
                            if(rsl_serv < RSLthresh):
                                ''' remove the channel utilisation for the user from serving sector # [SECTOR_METRIC_A]:  -- No. of channels currently in use'''
                                update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),1)
                                '''#move the call to archived list'''
                                update_user_list(param_opn.DELETE,user_index)
                                '''[USER_METRIC_E]:Call is reset for call setup of the user ; [USER_METRIC_F]:reset of the serving sector ID for the user ; # [USER_METRIC_G]:  reset Update the user call time stamp '''
                                user_info[user_index][3:7] = 0
                            else:
                                  pass                                                      
                    else: #else If the RSL neighbor < RSLserving + HO_m
                       
                        if(rsl_serv < RSLthresh):
                            '''#[SECTOR_METRIC_F]: record the call drop due to low signal strength (serving sector)'''
                            update_sector_metric(5,int(user_info[user_index][5]),serving_sector_num,0)    
                            ''' [SECTOR_METRIC_A]: remove the channel utilisation for the user from serving sector'''
                            update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),1)
                            '''#move the call to archived list'''
                            update_user_list(param_opn.DELETE,user_index) ##[SECTOR_METRIC_I]: -- remove this user under active call list    #[SECTOR_METRIC_J]: Move this user to archived call list
                            '''[USER_METRIC_E]:Call is reset for call setup of the user; [USER_METRIC_F]:reset of the serving sector ID for the user; [USER_METRIC_G]:  reset Update the user call time stamp '''
                            user_info[user_index][3:7] = 0
                        else:
                            pass
                #step 2 : Calculate the direction and next position
        ''' User mobility '''
        '''user is moving in North direction: Update user mobility'''
        if(user_info[user_index][1] == 1):
            distance_north = user_info[user_index][2]
            previous_distance_north = distance_north
            distance_north +=user_velocity
            
            if(distance_north <= upper_limit):
                user_info[user_index][2] = distance_north  
                if((user_info[user_index][4] == 1)):
                    si_value = rsl_serv - rsl_neigh
                    generate_si_list(si_value,user_index,int(user_info[user_index][5] ))
                user_info[user_index][7]+=1
            else:
                '''Active call in progress'''
                if((user_info[user_index][4] == 1)):
                    ''' # [SECTOR_METRIC_C]: ++ No. of successful calls(serving sector)'''
                    update_sector_metric(2,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                    '''# [SECTOR_METRIC_A]:  -- No. of channels currently in use'''
                    update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),1)
                    user_info[user_index][4] = 0
                ''' User re-positioning '''
                user_info[user_index][3:7] = 0
                new_value = np.random.uniform(lower_limit, upper_limit + 1)
                user_info[user_index][2] = new_value
                user_info[user_index][0] = new_value
                user_info[user_index][1] = np.random.randint(0,2)

                ''' Update the user mobility Distance - North direction'''
                if(user_info[user_index][1] == 1):
                    distance_north = user_info[user_index][2]
                    previous_distance_north = distance_north
                    distance_north +=user_velocity
                    if(distance_north <= upper_limit):
                        user_info[user_index][2] = distance_north  
                else:
                    distance_south = user_info[user_index][2]
                    previous_distance_south = distance_south
                    distance_south -=user_velocity
                    if(distance_south >= lower_limit):    
                        user_info[user_index][2]= distance_south       
        else:
            '''user is moving in South direction: Update user mobility''' 
            distance_south = user_info[user_index][2]
            previous_distance_south = distance_south
            distance_south -=user_velocity
            if(distance_south >= lower_limit):    
                user_info[user_index][2]= distance_south
                if((user_info[user_index][4] == 1) and ((user_info[user_index][7]>0) and (user_info[user_index][7]%7==0))):
                    si_value = rsl_serv - rsl_neigh
                    generate_si_list(si_value,user_index,int(user_info[user_index][5] ))
                user_info[user_index][7]+=1
            else:
                ''' Check active call '''
                if((user_info[user_index][4] == 1)):
                    ''' # [SECTOR_METRIC_C]: ++ No. of successful calls(serving sector)'''
                    update_sector_metric(2,int(user_info[user_index][5]),int(user_info[user_index][5]),0)
                    user_info[user_index][4] = 0 # reset of the ongoing call setup
                    '''# [SECTOR_METRIC_A]:  -- No. of channels currently in use'''
                    update_sector_metric(0,int(user_info[user_index][5]),int(user_info[user_index][5]),1)
                ''' User re-positioning '''
                new_value = np.random.uniform(lower_limit, upper_limit + 1)
                user_info[user_index][3:7] = 0
                user_info[user_index][2] = new_value
                user_info[user_index][0] = new_value
                user_info[user_index][1] = np.random.randint(0,2)
            
                if(user_info[user_index][1] == 1): #''' User moves in North direction '''
                    distance_north = user_info[user_index][2]
                    previous_distance_north = distance_north
                    distance_north +=user_velocity
                    if(distance_north <= upper_limit):
                        user_info[user_index][2] = distance_north  
                else:
                    distance_south = user_info[user_index][2]
                    previous_distance_south = distance_south
                    distance_south -=user_velocity
                    if(distance_south >= lower_limit):    
                        user_info[user_index][2]= distance_south
                
        user_index+=1 #index increment for user index'''
    time_index+=1     #index increment for timer index'''

''' Generate index values of User co-ordinate positions'''
start_points = np.arange(road_start, road_end, interval)
''' Generate Charts for SI distribution of alpha and beta sector '''
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(alpha_si_green_list,width=100,color="green")
plt.title("ALPHA SECTOR - SI Values Distribution- Green List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.grid(True)
plt.xticks(start_points, rotation=45)
plt.subplot(3, 1, 2)
plt.hist(alpha_si_magenta_list,width=100,color="magenta")
plt.title("ALPHA SECTOR - SI Values Distribution- Magenta List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.xticks(start_points, rotation=45)
plt.grid(True)
plt.subplot(3, 1, 3)
plt.hist(alpha_si_red_list,width=100,color="red")
plt.title("ALPHA SECTOR - SI Values Distribution- Red List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.xticks(start_points, rotation=45)
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(beta_si_green_list,width=100,color="green")
plt.title("BETA SECTOR - SI Values Distribution- Green List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.xticks(start_points, rotation=45)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.hist(beta_si_magenta_list,width=100,color="magenta")
plt.title("BETA SECTOR - SI Values Distribution- Magenta List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.xticks(start_points, rotation=45)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.hist(beta_si_red_list,width=100,color="red")
plt.title("BETA SECTOR - SI Values Distribution- Red List")
plt.xlabel("User co-ordinate positions")
plt.ylabel("Number of S/I points")
plt.xticks(start_points, rotation=45)
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
end_time = time.time()
execution_time = end_time - start_time
''' Display the SECTOR table info '''
print("\n\n\nvalue of sector metrics at the end of 4hrs")
print(f"{'No. of Users':<20}{num_users:<10}{'Road Length (in Km)':<20}{test_len:<10}"
      f"{'No. of Hrs Simulation':<25}{time_duration/3600:<10.2f}"
      f"{'Simulation Execution Time:':<25}{execution_time:.2f} seconds")
display_sector_table()
print("PART2 SIMULATION Completed ")




