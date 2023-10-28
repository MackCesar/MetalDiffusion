import tensorflow as tf
import subprocess
import re

def listDevices(
        module = "TensorFlow",
        printDevices = False
):
    """
    List available GPUs based on the machine learning module, defaults to TensorFlow
    """
    GPUs = []
                                  
    if module == "TensorFlow" or None:
        GPUs = tf.config.list_physical_devices("GPU")
    
    # First, find all GPU's, regardless of Metal compatability
    # Let's use system profiler in the Mac environment
    output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"])
    output = output.decode("utf-8")
    
    # Let's then parse the information from the profiler to find the GPUs
    deviceList = []
    current_gpu = {}
    for line in output.split('\n'):
        if line.strip() == '':
            continue
        if 'Chipset Model:' in line:
            if current_gpu:
                deviceList.append(current_gpu)
                current_gpu = {}
            current_gpu['name'] = line.split('Chipset Model:')[1].strip()
        elif 'VRAM' in line:
            current_gpu['vram'] = line.split('VRAM')[1].strip()

            # Slim down VRAM info to just the number, assuming gigabytes
            VRAMnumber = re.search(r'\d+', current_gpu['vram']).group()
            VRAMnumber = int(VRAMnumber)
            current_gpu['vram'] = VRAMnumber

        elif 'Vendor:' in line:
            current_gpu['vendor'] = line.split('Vendor:')[1].strip()
        elif 'Device ID:' in line:
            current_gpu['device_id'] = line.split('Device ID:')[1].strip()
        elif 'Metal Support:' in line:
            current_gpu['metalSupport'] = line.split('Metal Support:')[1].strip()
    if current_gpu:
        deviceList.append(current_gpu)
    
    # Now, remove Intel GPUs, which aren't currently supported in TensorFlow Metal
    for gpu in deviceList:
        if "Intel" in gpu['name']:
            deviceList.remove(gpu)
    
    # Finally, connect what Tensorflow found with what system profiler found, assuming they're discoverd in the same order
    i = 0
    for gpu in deviceList:
        try:
            gpu['TensorFlow'] = GPUs[i]
        except Exception as e:
            print(e)
            gpu['TensorFlow'] = GPUs[0]
        i += 1
    
    # Experimental, setting memory limits
    """try:
        for gpu in deviceList:
            #tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu['TensorFlow'],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2000)])
    except RuntimeError as e:
        print(e)"""
    
    """
    List available CPUs based on the machine learning module, defaults to TensorFlow
    """
    output = subprocess.check_output(["system_profiler", "SPHardwareDataType"])
    output = output.decode("utf-8")

    currentCPU = {}
    for line in output.split('\n'):
        if line.strip() == '':
            continue
        if 'Processor Name:' in line:
            if currentCPU:
                deviceList.append(currentCPU)
                currentCPU = {}
            currentCPU['name'] = line.split('Processor Name:')[1].strip()
        elif 'Processor Speed:' in line:
            currentCPU['speed'] = line.split('Processor Speed:')[1].strip()
        elif 'Number of Processors:' in line:
            currentCPU['num_processors'] = line.split('Number of Processors:')[1].strip()
        elif 'Total Number of Cores:' in line:
            currentCPU['num_cores'] = line.split('Total Number of Cores:')[1].strip()
    if currentCPU:
        currentCPU['TensorFlow'] = tf.config.list_physical_devices("CPU")[0]
        deviceList.append(currentCPU)

    if printDevices is True:
        print(deviceList)

    # MacOS 14 bug: CPU dictionary object does NOT have a 'name' key,
    # and this key is required later in dream.py. Might be not only the MacOS problem,
    # therefore: remove devices with no 'name'.
    deviceList = list(filter(lambda device: 'name' in device, deviceList))

    return deviceList
