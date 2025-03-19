import os
from os.path import expanduser
import numpy as np
import tarfile, re, subprocess, sys, yaml
from linecache import getline
from clawpack.geoclaw import topotools as topo


#####################
# various functions #
#####################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Function to flatten a nested dictionary
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def format_numbers(features):
    """ 
    Format numbers to two decimal places if they are floats
    """
    formatted_features = []
    for feature in features:
        name, value = feature
        if isinstance(value, float):  # Check if the value is a float
            formatted_value = f"{value:>10.2f}"  # Format to two decimal places and right-align
        else:
            formatted_value = f"{value:>10}"  # Right-align non-float values
        formatted_features.append([name, formatted_value])
    return formatted_features

############
# clawpack #
############
 

# check if clawpack is installed
def check_claw():
    """ 
    Test if clawpack is installed. If so, it returns the CLAW path
    """
    CLAW = os.environ['CLAW']
    home = expanduser("~")
    if CLAW=='':
        claw = False
        with open(home+'/.bashrc') as f:
            datafile = f.readlines()
        for line in datafile:
            s="CLAW"
            if s in line and line.find('#')==-1:
                claw=(str.split(line))[1]
                claw = home+claw.replace("CLAW=$HOME", "")
                return claw
        if not claw:
            print("Error: I cannot determine the $CLAW variable...")
            print("Please modify the script and define it explicitely")
            return claw
    else:
        return CLAW
    
def check_version(claw):
    claw_setup = claw+"/setup.py"
    # Open the file
    with open(claw_setup, "r") as file:
        # Initialize variables to store MAJOR and MINOR values
        major_value = None
        minor_value = None
        # Iterate through each line in the file
        for line in file:
            # Strip leading/trailing whitespace
            line = line.strip()
            # Check if the line starts with "MAJOR" and contains "="
            if line.startswith("MAJOR") and "=" in line:
                # Split the line by "=" and extract the value
                parts = line.split("=")
                if len(parts) == 2:                      # Ensure the line is properly formatted
                    major_value = int(parts[1].strip())  # Extract and convert to integer
            # Check if the line starts with "MINOR" and contains "="
            elif line.startswith("MINOR") and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:   
                    minor_value = int(parts[1].strip())  #  
    return [major_value,minor_value]


def get_version_from_file(file_path):
    """Extract the version from a file if it exists in the working directory."""
    # Regular expression to extract filename & version number
    pattern = r"[#!]\s*([\w\.]+).*?version\s*=\s*([\d\.]+)"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            match = re.search(pattern, first_line)
            if match:
                return match.group(1), float(match.group(2))  # Return (filename, version)
    except Exception:
        pass  # Ignore errors
    return None, None


def install_avac(**kwargs):
    """ 
    This function extracts the files required by AVAC
    options:
    - verbosity = False by default, makes the execution verbose or not
    - path = '.' by defaut (working directory). If a new path is specified
      and does not exist, it is created
    - archive = 'files.tar.gz' by default
    """
    path = kwargs.get('path', '.')
     
    if not isinstance(path,str): 
        print(f"Error. Check your path definition! You set path = {path}.")
    else:
        if os.path.isdir(path):
            print(f"Installation of AVAC in the working directory: {os.getcwd()}")
        else:
            print(f"Installation of AVAC in the new directory {os.getcwd()+'/'+path}")
            os.makedirs(path)
    
    archive = kwargs.get('archive', 'files.tar.gz')
    verbosity = kwargs.get('verbosity', False)
      
    #extract_path = '.'  # Extract to current directory
    if not os.path.isfile(archive): 
        print(f"Installation impossible. Archive {archive} is missing.")
        print(f"Stopped...")
    else:
        # Regular expression to extract filename & version number
        pattern = r"[#!]\s*([\w\.]+).*?version\s*=\s*([\d\.]+)"

        # Open the tar.gz archive
        with tarfile.open(archive, "r:gz") as tar:
            file_names = tar.getnames()

            for target_file in file_names:
                extracted_file = tar.extractfile(target_file)

                if extracted_file:  # Ignore directories                    
                    # Read first line separately to extract version
                    first_line = extracted_file.readline().decode().strip()
                    match = re.search(pattern, first_line)

                    if match:
                        filename = match.group(1)  # Extract filename
                        archive_version = float(match.group(2))  # Extract version
                        file_path = os.path.join(path, filename)

                        # Read the remaining content after the first line
                        remaining_content = extracted_file.read()

                        # Check if file exists in the working directory
                        if os.path.exists(file_path):
                            _, existing_version = get_version_from_file(file_path)
                            
                            if existing_version is None or existing_version < archive_version:
                                if verbosity:
                                    print(f"Updating {filename} (Old: {existing_version}, New: {archive_version})")
                                with open(file_path, "wb") as f_out:
                                    f_out.write((first_line + "\n").encode())  # Restore first line
                                    f_out.write(remaining_content)  # Write the rest
                            else:
                                if verbosity:
                                    print(f"Skipping {filename}, version {existing_version} is up to date.")
                        else:
                            if verbosity: print(f"Extracting new file: {filename}")
                            with open(file_path, "wb") as f_out:
                                f_out.write((first_line + "\n").encode())  # Restore first line
                                f_out.write(remaining_content)  # Write the rest
    print(f"=> You are using AVAC version {get_version_from_file('Makefile')[1]}.")



# running AVAC


def make_output(avac_p,verbosity=True):
    """
    Execute Make clean, then Make :output
    Input: verbosity (Boolean): If True, displays messages during execution; otherwise,
                            directs the messages to the file 'avac.log'.
    Output: output files in _output, avac.log.
    """
    if not isinstance(verbosity, bool):
        verbosity = True
    
    print(f"I will make an AVAC computation.")
    tmax = avac_p['computation']['t_max']
    dt = tmax / avac_p['computation']['nb_simul']
    print(f"Times: from t = 0 to t = {tmax} s with a time step dt = {dt} s.")

    subprocess.run(["make","clean" ])
    # Run the command based on verbosity
    if not verbosity:
        # Suppress stdout and capture stderr
        result = subprocess.run(["make", ".output"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    else:
        # Capture both stdout and stderr
        result = subprocess.run(["make", ".output"], capture_output=True, text=True)

    # Save the output to a file
    with open("avac.log", "w") as f:
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)

    # Display output in the notebook based on verbosity
    if verbosity:
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

    # Check if the execution was successful
    if result.returncode == 0:
        print(f"Computation is successful.")
    else:
        print("Failed! See the log file: avac.log.")


# animation
def make_animation(avac_p,verbosity=True):
    """
    Execute the animation script make_fgout_animation.py
    Input: verbosity (Boolean): If True, displays messages during execution; otherwise,
                            directs the messages to the file 'animation.log'.
    Output: mp4 and html files, animation.log.
    """
    if not isinstance(verbosity, bool):
        verbosity = True
    print(f"I will make an animation for the {avac_p['animation']['variable']} variable.")
    tmax = avac_p['computation']['t_max']
    dt = tmax / avac_p['animation']['n_out']
    print(f"Times: from t = 0 to t = {tmax} s with a time step dt = {dt} s.")
    # Open the log file for writing
    with open("animation.log", "w") as log_file:
        # Start the process
        process = subprocess.Popen(
            ["make", "animation"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            universal_newlines=True
        )
        # Read and display output in real-time
        while True:
            # Read stdout line by line
            stdout_line = process.stdout.readline()
            if stdout_line:
                if verbosity:
                    print(stdout_line, end="")  # Display in real-time
                log_file.write(stdout_line)  # Write to log file
            # Read stderr line by line
            stderr_line = process.stderr.readline()
            if stderr_line:
                if verbosity:
                    print(stderr_line, end="", file=sys.stderr)  # Display in real-time
                log_file.write(stderr_line)  # Write to log file
            # Check if the process has finished
            if process.poll() is not None:
                break
        # Ensure all remaining output is captured
        for stdout_line in process.stdout:
            if stdout_line:
                if verbosity:
                    print(stdout_line, end="")
                log_file.write(stdout_line)
        for stderr_line in process.stderr:
            if stderr_line:
                if verbosity:
                    print(stderr_line, end="", file=sys.stderr)
                log_file.write(stderr_line)
        # Wait for the process to complete and get the return code
        return_code = process.wait()
    # Check if the execution was successful
    if return_code == 0:
        print(f"Creation of the mp4 file: AVAC_animation_for_{avac_p['animation']['variable']}.mp4 in the working directory.")
        print(f"Creation of the html file: AVAC_animation_for_{avac_p['animation']['variable']}.html in the working directory.")
    else:
        print("Failed! See the log file: animation.log.")
    
###################
# post-processing #
###################
fn_eta      = lambda q: q[3,:,:]                     # eta = z_b +h
fn_sol      = lambda q: q[3,:,:] - q[0,:,:]          # z_b
fn_h        = lambda q: q[0,:,:]                     # h
fn_husquare = lambda q: q[1,:,:]**2+q[2,:,:]**2      # h²(u²+v²)
fn_extract  = lambda q: array((fn_h(q),fn_eta(q))) # (h, eta)
fn_hu       = lambda q: q[1,:,:]                     # hu
fn_hv       = lambda q: q[2,:,:]                     # hv
fn_u        =  lambda q: np.where(q[0,:,:]>0, (q[1,:,:]/q[0,:,:]), 0)  # u
fn_v        =  lambda q: np.where(q[0,:,:]>0, (q[2,:,:]/q[0,:,:]), 0)  # v    
fn_velocity =  lambda q: np.where(q[0,:,:]>0, np.sqrt((q[2,:,:]/q[0,:,:])**2+(q[1,:,:]/q[0,:,:])**2), 0)  # v 


######################
# initial conditions #
######################

def correctingFactor1(s,theta,nu):
    """
    De Quervain's correction of d_0
    Input: local slope, theta = critical slope, nu: coefficient
    Output: correction
    """
    theta_rad = theta/180*np.pi # conversion to radians
    q         = np.arctan(s)    # conversion from slope to angle (radians)
    if q>25*np.pi/180.:
       return (np.sin(theta_rad)-nu*np.cos(theta_rad))/(np.sin(q)-nu*np.cos(q))
    else:
       return 0

def correctingFactor2(z,zref,gradient_hypso):
    """
    Burkard's correction of d_0
    Input: local elevation, zref: elevation of the measurement station
           gradient_hypso: hypsometric gradient (additional snow [m] quantity per 100-m altitude range)
    Output: correction
    """
    return (z-zref)*gradient_hypso/100

##########
# raster #
##########
def extract_values(text):
    """ 
    Goal: extracting the number and word from a string 
    Output: Boolean, number, word, remark
    The Boolean is True when extraction is successful, False otherwise.
    The remark is a text generated when something unusual is met
    """
    # Regular expression to find the first number
    number_pattern = r'-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'  # Matches integers, floats, and scientific notation
    # Search for the first number
    numbers = re.findall(number_pattern, text)
    num_count = len(numbers)
    number_match = numbers[0]
    # Search for the first word
    word_pattern = r'\b[a-zA-Z_]+\b'  # Matches any word (letters only and underscore)
    # Regular expression to find the first word
    word_match = re.search(word_pattern, text)
    if word_pattern == 'cellsize':
        if num_count > 1:
            remark = "rectangular cells"
        else:
            remark = "square cells"
    else:
        if num_count > 1:
            remark = "more than one value for "+word_match.group()
        else:
            remark = ""
    if number_match and word_match:
        number = number_match # Extract number
        word = word_match.group()
        return True, number,word,remark
    else:
        return False

def count_header_lines(filepath, num_lines=10):
    """
    dertermines the header size of the raster file, i.e.
    the number of lines with alphanumeric information
    Input: file name
    Output: number of lines
    """
    count = 0
    for i in range(1, num_lines + 1):  # Les lignes dans linecache commencent à 1
        line = getline(filepath, i).strip()  # Supprime espaces et \n
        
        # Supprime les nombres (y compris en notation scientifique) au début de la ligne
        cleaned_line = re.sub(r'^[\s\d\.\-+eE]+', '', line).strip()

        # Vérifie s'il reste au moins une lettre dans la ligne
        if re.search(r'[a-zA-Z]', cleaned_line):
            count += 1

    return count

def determine_file_type(file):
    """ 
    Goal: determining the nature of a raster file
    Input: raster *.brage.asc
    Output: the file type (grass, esri or claw format)
    """
    try:
        with open(file, "r") as file:
            text = file.readline().strip()
        # Regular expression to find the first number
        number_pattern = r'-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'  # Matches integers, floats, and scientific notation
        # Search for the first number
        number_match = re.search(number_pattern, text)
        # Search for the first word
        word_pattern = r'\b[a-zA-Z_]+\b'  # Matches any word (letters only)
        # Regular expression to find the first word
        word_match = re.search(word_pattern, text)
        start_position = word_match.start() 
        word = word_match.group()
        type_file = 'esri'
        if start_position == 0:
            type_file = 'esri'
        else:
            type_file = 'claw'
        if word in ['north','south','east','west']:
            type_file = 'grass'
        
        if type_file in ['esri','claw','grass']:
            return type_file
        else:
            print(f'Error!I cannot determine the type of the file {file}.')
    except FileNotFoundError:
        print(f"The file '{file}' does not exist.")

# export-to-Qgis function 
def export_raster(fname,tableau,xll,yll,cellsize,ndata=-9999):
    """
    export numpy arrays 'tableau' to file fname in an esri ASCII format
    """
    header =  "ncols        %s\n" % tableau.shape[0]
    header += "nrows        %s\n" % tableau.shape[1]
    header += "xllcorner    %s\n" % xll
    header += "yllcorner    %s\n" % yll
    header += "cellsize     %s\n" % cellsize
    header += "nodata_value %s\n" % ndata
    np.savetxt(fname, np.nan_to_num(tableau.T[::-1,:] ,nan=ndata), header=header, fmt="%1.2f", comments='')

#################################
# importing raster and shapefiles
def reading_raster_file(source, nan_replace = False): 
    '''
    Read raw data from source. The source uses ASCII Grass format (based on cardinal directions). The 
    function reading_raster_file does some work to read and convert these data
    into a format compatible with clawpack
    For more information, see https://www.clawpack.org/grid_registration.html#grid-registration 
    '''
    hdr_size = count_header_lines(source, num_lines=10)  # header size
    tab = np.genfromtxt(source, skip_header=hdr_size, missing_values='*' ) 
    if nan_replace: tab = np.nan_to_num(tab,nan=-9999)
    
    hdr = [getline(source, i) for i in range(1, hdr_size+1)]
     
    header_extraction = np.array([extract_values(string) for string in hdr])
    values = [float(val) for val in header_extraction[:,1]]
    keys = header_extraction[:,2]
    type_file = determine_file_type(source)
    if 'xllcenter' in keys:
        grid_type = 'grid'
    else:
        grid_type = 'cell'
    dictionnaire = {keys[k]:values[k] for k in range(0,hdr_size)}
    # DEM extent
    if (type_file == 'grass'):
        ymin = dictionnaire['south']
        ymax = dictionnaire['north']
        xmin = dictionnaire['west']
        xmax = dictionnaire['east']
        nbx  = int(dictionnaire['cols'])
        nby  = int(dictionnaire['rows'])
    if (grid_type == 'cell') and (type_file == 'esri'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['yllcorner']
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xllcorner']
        xmax = xmin+nbx*cell_size
    if (type_file == 'claw'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['ylower']
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xlower']
        xmax = xmin+nbx*cell_size
    if (grid_type == 'grid') and (type_file == 'esri'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['yllcenter']-cell_size/2
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xllcenter']-cell_size/2
        xmax = xmin+nbx*cell_size
 
    x = np.linspace(xmin,xmax,nbx)
    y = np.linspace(ymin,ymax,nby)
    X_fine_grid, Y_fine_grid = np.meshgrid(x,y)

    init = topo.Topography()
    init.X = X_fine_grid
    init.Y = Y_fine_grid 
    init.Z = tab[::-1,:]
    init.y = Y_fine_grid[:,0]
    init.x = X_fine_grid[0,:]
    return init

def reading_raster_file_features(source): 
    '''
    Read raster data from source. The source uses ASCII Grass format (based on cardinal directions). The 
    function reading_raster_file does some work to read and convert these data
    into a format compatible with clawpack
    For more information, see https://www.clawpack.org/grid_registration.html#grid-registration 
    input: raster file
    output: xmin, xmax, ymin, ymax, nbx, nby, cell_size, dictionnaire, failure, remarks
    '''
    hdr_size = count_header_lines(source, num_lines=10)  # header size
    hdr = [getline(source, i) for i in range(1, hdr_size+1)]
    header_extraction = np.array([extract_values(string) for string in hdr])
    values = [float(val) for val in header_extraction[:,1]]
    keys = header_extraction[:,2]
    type_file = determine_file_type(source)
    remarks = header_extraction[:,3]
    failure = header_extraction[:,0]
    if 'xllcenter' in keys:
        grid_type = 'grid'
    else:
        grid_type = 'cell'
    dictionnaire = {keys[k]:values[k] for k in range(0,hdr_size)}
    # DEM extent
    if (type_file == 'grass'):
        ymin = dictionnaire['south']
        ymax = dictionnaire['north']
        xmin = dictionnaire['west']
        xmax = dictionnaire['east']
        nbx  = int(dictionnaire['cols'])
        nby  = int(dictionnaire['rows'])
        cell_size = (xmax-xmin)/nbx
    if (grid_type == 'cell') and (type_file == 'esri'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['yllcorner']
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xllcorner']
        xmax = xmin+nbx*cell_size
    if (type_file == 'claw'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['ylower']
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xlower']
        xmax = xmin+nbx*cell_size
    if (grid_type == 'grid') and (type_file == 'esri'):
        nbx  = int(dictionnaire['ncols'])
        nby  = int(dictionnaire['nrows'])
        cell_size = dictionnaire['cellsize']
        ymin = dictionnaire['yllcenter']-cell_size/2
        ymax = ymin+nby*cell_size
        xmin = dictionnaire['xllcenter']-cell_size/2
        xmax = xmin+nbx*cell_size
    dictionary_extent = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'nbx':nbx,'nby':nby,'cell_size':cell_size,'nodata_value':-9999}

    return xmin, xmax, ymin, ymax, nbx, nby, cell_size, dictionary_extent, failure, remarks, grid_type 

def check_raster(file):
    """
    checks whether 'file' is a raster file
    Output: True if the file is a raster (or no error has been pinpointed)
            False if import raises problems
    """
    print(f"Raster file: {file}") 
    print()
    if os.path.isfile(file):
        print(f"File {file} exists in the wording directory.")
    print()
    xmin, xmax, ymin, ymax, nbx, nby, cell_size, dico, failure, remarks, grid_type  = \
          reading_raster_file_features(file)
    raster_features = [['xmin',xmin],['xmax',xmax],['ymin',ymin],['ymax',ymax],['nbx',nbx],['nby',nby],['cell size',cell_size]]
    # Check if all strings are empty
    test_all_remark_empty   = np.all(remarks == '')
    test_all_success_import = np.all(failure == 'True')
    if test_all_remark_empty and test_all_success_import:
        print("No problem detected in the raster file")
    elif test_all_success_import:
        # some problems detected
        non_empty_remark = remarks[remarks != '']
        print(f"I detected {len(non_empty_remark )} potential problem(s):")
        for rmk in non_empty_remark:
            print("* ", rmk)
    else:
        print("Check your file! I am not able to import it as a raster file.")
    raster_type = determine_file_type(file)
    print()
    print('Raster features')
    print(f"* The raster format is: {raster_type}.")   
    print(f"* The grid type is: {grid_type}.")      

    # Format the numbers in raster_features
    formatted_raster_features = format_numbers(raster_features)

    # Calculate column widths
    col_width_feature = max(len(row[0]) for row in formatted_raster_features)
    col_width_value = max(len(row[1]) for row in formatted_raster_features)

    # Print the table with custom formatting
    # Print headers
    print()
    print("-" * (col_width_feature + col_width_value + 3))
    print(f"{'Feature':<{col_width_feature}} {'Value':>{col_width_value}}")
    print("-" * (col_width_feature + col_width_value + 3))

    # Print rows
    for feature, value in formatted_raster_features:
        print(f"{feature:<{col_width_feature}} {value:>{col_width_value}}")

    if test_all_success_import: 
        return True
    else:
        return False
    
def export_claw_dem(name_file,xmin,xmax,ymin,ymax,nbx,nby,alt):
    """
    convert the DEM to a claw format and save it
    """
    print(f'Export of DEM to file {name_file}.')
    x = np.linspace(xmin,xmax,nbx)
    y = np.linspace(ymin,ymax,nby)
    X_fine_grid, Y_fine_grid = np.meshgrid(x,y)

    init = topo.Topography()
    init.X = X_fine_grid
    init.Y = Y_fine_grid 
    init.Z = alt
    init.y = Y_fine_grid[:,0]
    init.x = X_fine_grid[0,:]
    init.write(name_file,topo_type=3)

def export_claw_initiation_file(topo_file,zi):   
    """
    save the initiation file
    """
    print(f'Export of initial conditions to file init.xyz.')   
    init   = topo.Topography()
    init.X = topo_file.X 
    init.Y = topo_file.Y 
    init.Z = zi[:,:] 
    init.y = init.Y[:,0]
    init.x = init.X[0,:]
    init.write('init.xyz',topo_type=1)
    print(f'* maximum initial depth of starting zone  = {np.max(zi[:,:])} m')    

def test_keys(dictionary_tested,dictionary_ref):
    """ 
    checks whether all the keys are defined in the loaded dictionary 
    """
    if not isinstance(dictionary_ref, dict):
        dictionary_ref = {key: None for key in dictionary_ref}
    if not sorted(list(dictionary_tested.keys()) )==sorted(list(dictionary_ref.keys()) ):
        print('The configuration file does not satisfy the requirements.')
        difference_1 = set(dictionary_ref.keys())-set(dictionary_tested.keys())
        difference_2 = set(dictionary_tested.keys())-set(dictionary_ref.keys())
        if len(difference_1)>0:
            print(f"There is missing information. Please check your configuration file. Computation will fail otherwise!")
            for info in difference_1: print(f"* Missing key: {info}")
            return 1
        if len(difference_2)>0:
            print(f"There is additional information. This information will not be used here.")
            for info in difference_2: print(f"* Additional key: {info}")
            return 0
    else:
        return 0

###########
# testing #
###########

def import_configuration_files(file_name):
    """
    import the configuration and check consistency
    output: a dictionary with AVAC parameters
    the script can pinpoint potential errors in the parameters
    """
    print(f"Opening the configuration file {file_name}...")

    with open(file_name, 'r') as file:
        avac_parameters = yaml.safe_load(file)

    def is_genuine_int(var):
        return isinstance(var, int) and not isinstance(var, bool)    

    def is_integer(key_1,key_2):
        """check whether avac_parameters[key_1][key_2] is integer"""
        var = avac_parameters[key_1][key_2]
        if not is_genuine_int(var):
            print(f"The variable {key_1}.{key_2} must be an integer! Here, I get {key_1}.{key_2} = {var}")
            return 1
        else:
            return 0

    def is_boolean(key_1,key_2):
        """checks whether avac_parameters[key_1][key_2] is boolean"""
        var = avac_parameters[key_1][key_2]
        if not isinstance(var, bool):
            print(f"The variable {key_1}.{key_2} must be a Boolean! Here, I get {key_1}.{key_2} = {var}")
            return 1
        else:
            return 0

    keys_avac          = ['computation', 'release', 'rheology', 'topography','output','animation']
    keys_computation   = ['cfl_max', 'cfl_target', 'dry_limit', 'max_iter', 'nb_simul', \
                        'refinement', 't_max','domain_cell','boundary','output_directory']
    keys_release       = ['correction_elevation', 'correction_slope', 'd0', 'gradient_hypso', 'nu', 'theta_cr', 'z_ref']
    keys_rheology      = ['beta', 'model', 'mu', 'rho', 'u_cr', 'xi']
    keys_topography    = ['dem', 'starting_areas']    
    keys_output        = ['output_format', 'verbosity', 'delta_t']
    keys_animation     = ['n_out','variable']
    rheological_models = ['Voellmy','Coulomb']
    output_formats     = ['ascii','binary32','binary64']
    verbosity_formats  = [0,1,2,3,4] # see https://www.clawpack.org/pyclaw/output.html
    boundary_formats   = ['wall','extrap','user']
    animation_formats   = ['pressure','depth','velocity']

    error = 0
    error += test_keys(avac_parameters,keys_avac) # check whether the config file keys are those expected

    print()
    # Checs topography
    error += test_keys(avac_parameters['topography'],keys_topography)
    file_path = avac_parameters['topography']['dem']
    if os.path.isfile(file_path):
        print(f"- I found the DEM file {file_path}.")
    test_success = [bool(chain) for chain in reading_raster_file_features(file_path)[8] ]
    if np.all(np.array(test_success)):
        print("  File import raises no issue.")
    else:
        print(f"  When importing file {avac_parameters['topography']['dem']}, I found errors in the header. Please check.")
        error += 1
    file_path = avac_parameters['topography']['starting_areas']
    if os.path.isfile(file_path):
        print(f"- I found the shapefile {file_path} containing the starting areas.")
        print(f"  It seems ok.")
    else:
        print(f"- I failed to import {file_path}! Please check.")
        error += 1
    file_path = avac_parameters['topography']['starting_areas'][:-3]+'shx'
    if not os.path.isfile(file_path):
        print(f"- File {file_path} is missing! Please check. ")
        print(f"  This file accompanies the shapefile. Find it or reconstruct it using gdal. ")
        error +=1
    # checks output
    error += test_keys(avac_parameters['output'],keys_output)
    error += is_integer('output','verbosity')
    if avac_parameters['output']['output_format'] not in output_formats:
        print(f"The output format {avac_parameters['output']['output_format'] } is unknown!")
        print("The only current possibilities are: 'ascii', 'binary64' or 'binary32'.")
    if avac_parameters['output']['verbosity'] not in verbosity_formats:
        print(f"The verbosity parameter is set to {avac_parameters['output']['verbosity'] }.")
        print("It should range from 0 to 4.")
    if avac_parameters['output']['delta_t'] > avac_parameters['computation']['t_max']:
        print(f"The parameter delta_t is set to {avac_parameters['output']['delta_t'] }.")
        print(f"It is larger to t_max = {avac_parameters['computation']['t_max'] }!")
        print("I correct it. Check!")
        avac_parameters['output']['delta_t'] = avac_parameters['computation']['t_max']
    # checks computation parameters
    error += test_keys(avac_parameters['computation'],keys_computation)
    if (avac_parameters['computation']['cfl_max']>1) or (avac_parameters['computation']['cfl_max']<0):
        print(f"Check variable cfl_max = {avac_parameters['computation']['cfl_max']}")
        print(f"This value should be an integer in the 0.1-1 range.")
        error += 1
    if (avac_parameters['computation']['cfl_target']>avac_parameters['computation']['cfl_max']):
        print(f"Check variable cfl_target = {avac_parameters['computation']['cfl_max']}")
        print(f"This value cannot be larger car cfl_max = {avac_parameters['computation']['cfl_max']}.")
        error += 1
    error += is_integer('computation','max_iter')
    error += is_integer('computation','nb_simul')
    error += is_integer('computation','refinement')
    if (avac_parameters['computation']['refinement']<1) or (avac_parameters['computation']['refinement']>6):
        print(f"Check variable refinement = {avac_parameters['computation']['refinement']}.")
        print(f"This value should be an integer in the 1-6 range.")
    if avac_parameters['computation']['boundary'] not in boundary_formats:
        print(f"The boundary condition {avac_parameters['computation']['boundary'] } is unknown!")
        print("The only current possibilities are: 'ascii', 'binary64' or 'binary32'.")
    if avac_parameters['computation']['boundary'] == 'user':
        print("This is not implemented by default. Check file bc2.f90.")
    # check release parameters
    error += test_keys(avac_parameters['release'],keys_release)
    error += is_boolean('release','correction_elevation')
    error += is_boolean('release','correction_slope')
    if (avac_parameters['release']['gradient_hypso']<0) or (avac_parameters['release']['gradient_hypso']>0.2):
        print(f"Check variable gradient_hypso = {avac_parameters['release']['gradient_hypso']}.")
        print(f"This value cannot be negative or larger than 20 cm/100 m.")
        error += 1
    if (avac_parameters['release']['z_ref']<0) or (avac_parameters['release']['z_ref']>9000):
        print(f"Check variable z_ref = {avac_parameters['release']['z_ref']}.")
        print(f"This value cannot be negative or larger than 9000 m.")
        error += 1
    if (avac_parameters['release']['theta_cr']<5) or (avac_parameters['release']['theta_cr']>50):
        print(f"Check variable theta_cr = {avac_parameters['release']['theta_cr']}.")
        print(f"This value should be expressed in degrees and close to 30.")
        error += 1
    # check animation
    error += test_keys(avac_parameters['animation'],keys_animation)
    error += is_integer('animation','n_out')
    if avac_parameters['animation']['variable'] not in animation_formats:
        print(f"The variable {avac_parameters['animation']['variable'] } for animation is unknown!")
        print("The only current possibilities are: 'pressure', 'depth' or 'velocity'.")
    # check rheology parameters
    error += test_keys(avac_parameters['rheology'],keys_rheology)
    if avac_parameters['rheology']['model'] not in rheological_models:
        print(f"The rheological model {avac_parameters['rheology']['model'] } is unknown!")
        print("The only current possibilities are: 'Voellmy' or 'Coulomb'.")
    if (avac_parameters['rheology']['mu']<0.05) or (avac_parameters['rheology']['mu']>0.5):
        print(f"Check variable mu = {avac_parameters['rheology']['mu']}.")
        print(f"This value should be in the 0.05-0.5 range.")
        error += 1
    if (avac_parameters['rheology']['xi']<100) or (avac_parameters['rheology']['xi']>1e4):
        print(f"Check variable xi = {avac_parameters['rheology']['xi']}.")
        print(f"This value should be in the 100-10,000 range.")
    if (avac_parameters['rheology']['u_cr']<0) or (avac_parameters['rheology']['u_cr']>0.5):
        print(f"Check variable u_cr = {avac_parameters['rheology']['u_cr']}.")
        print(f"This value should be in the 0-0.5 range.")
        error += 1
    if (avac_parameters['rheology']['rho']<100) or (avac_parameters['rheology']['rho']>1000):
        print(f"Check variable rho = {avac_parameters['rheology']['rho']}.")
        print(f"This value should be in the 100-1000 range.")
        error += 1
    if (avac_parameters['rheology']['beta']<0) or (avac_parameters['rheology']['u_cr']>1.5):
        print(f"Check variable beta = {avac_parameters['rheology']['beta']}.")
        print(f"This value should be in the 0-1.5 range.")
        error += 1
    print()    
    if error>0:
        print(f"Error(s) detected: {error}")
    else:
        print("Everything looks fine so far...")
    print()
    # Flatten the configuration dictionary
    flat_configuration = flatten_dict(avac_parameters)

    print("Configuration file:")
    for key, value in flat_configuration.items():
        var_name = key.replace('.', '_')
        globals()[var_name] = value
        print(f"* {var_name} = {value}")
    return avac_parameters
