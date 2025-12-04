import numpy as np
import matplotlib.pyplot as plt

class Port:

    def __init__(self, name, index = None):

        self.name = name
        self.index = index
    
    def __str__(self):
        return f"Port(name={self.name!r}, index={self.index})"



class BaseComponent:

    def __init__(self, name, input_names, output_names, transfer_matrix):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.transfer_matrix = transfer_matrix

    def __str__(self):
        return (f"Component(name={self.name!r}, "
                f"component type = {self.__class__.__name__}, "
                f"inputs={[p for p in self.input_names]!r}, "
                f"outputs={[p for p in self.output_names]!r}),"
                f"transfer_matrix={self.transfer_matrix!r})")

class DC(BaseComponent):

    def __init__(self, name, input_names, output_names, custom_matrix = None):


        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            transfer_matrix = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

        super().__init__(name, input_names, output_names, transfer_matrix)

class GeneralSplitter(BaseComponent):

    def __init__(self, name, input_names, output_names, split_N = 2, custom_matrix = None):

        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            # Default transfer matrix for General Splitter
            transfer_matrix = np.array([[1/np.sqrt(split_N)] * split_N] * split_N, dtype=complex)

        super().__init__(name, input_names, output_names, transfer_matrix)

class Ysplitter(BaseComponent):

    def __init__(self, name, input_names, output_names, split_ratio = 0.5, custom_matrix = None):

        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            # Default transfer matrix for Y-splitter
            transfer_matrix = np.array([[np.sqrt(split_ratio), np.sqrt(1 - split_ratio)]], dtype=complex)

        super().__init__(name, input_names, output_names, transfer_matrix)

class Ycombiner(BaseComponent):

    def __init__(self, name, input_names, output_names, combine_ratio = 0.5, custom_matrix = None):

        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            # Default transfer matrix for Y-combiner
            transfer_matrix = np.array([[np.sqrt(combine_ratio), np.sqrt(1 - combine_ratio)]], dtype=complex).T

        super().__init__(name, input_names, output_names, transfer_matrix)

class PhaseShifter(BaseComponent):
    
    def __init__(self, name, input_names, output_names, phase_shift = 0, custom_matrix = None):

        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            # Default transfer matrix for Phase Shifter
            transfer_matrix = np.array([[np.exp(1j * phase_shift)]], dtype=complex)

        super().__init__(name, input_names, output_names, transfer_matrix)


class TriCoupler(BaseComponent):

    def __init__(self, name, input_names, output_names, custom_matrix=None, use_test_tricoupler=False):
        
        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            if use_test_tricoupler:
                # an unitary matrix that enables coupling all the planet light into the central channel
                transfer_matrix = np.array([[1/2, 1/np.sqrt(2), -1/2],
                                            [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                                            [-1/2, 1/np.sqrt(2), 1/2]], dtype=complex)
            else:
                # Default transfer matrix for Tri-Coupler
                phase = 2*np.pi/3
                transfer_matrix = (1/np.sqrt(3)) * np.array([[1, np.exp(1j * phase), np.exp(1j * phase)],
                                                            [np.exp(1j * phase), 1, np.exp(1j * phase)],
                                                            [np.exp(1j * phase), np.exp(1j * phase), 1]], dtype=complex)

        super().__init__(name, input_names, output_names, transfer_matrix)

class PortSelector(BaseComponent):
    """Component that selects only specific ports from input ports"""
    
    def __init__(self, name, input_names, selected_port_names, custom_matrix=None):
        
        if custom_matrix is not None:
            transfer_matrix = custom_matrix
        else:
            # Create a selection matrix
            n_inputs = len(input_names)
            n_outputs = len(selected_port_names)
            
            transfer_matrix = np.zeros((n_inputs, n_outputs), dtype=complex)
            
            # Set 1.0 for selected ports
            for j, selected_name in enumerate(selected_port_names):
                if selected_name in input_names:
                    i = input_names.index(selected_name)
                    transfer_matrix[i, j] = 1.0
                    
        super().__init__(name, input_names, selected_port_names, transfer_matrix)


class PIC(BaseComponent):

    def __init__(self, input_port_names, output_port_names=None, name="Device"):
        
        # If no output ports specified, assume same as input ports
        if output_port_names is None:
            output_port_names = input_port_names.copy()
        
        self.input_port_names = input_port_names
        self.output_port_names = output_port_names
        self.inputs = [Port(name) for name in input_port_names]

        self.all_ports = self.inputs

        for i, p in enumerate(self.inputs):
            # assign port index
            p.index = i

        self.components = []
        self.global_matrices = []
        
        # Initialize as BaseComponent with current transfer matrix (identity initially)
        initial_transfer_matrix = np.eye(len(input_port_names), dtype=np.complex_)
        super().__init__(name, input_port_names, output_port_names, initial_transfer_matrix)
    
    def add_component(self, component, verbose = False):
        self.components.append(component)

        indices_0 = [p.index for p in self.all_ports]
        if verbose:
            print("current port indices", indices_0)

        if verbose:
            print("adding component", component.name)
            # print("inputs", component.input_names)
            # print("outputs", component.output_names)

        # Get all unique port names from component
        all_component_ports = set(component.input_names + component.output_names)
        
        # Create new ports for names that don't exist
        existing_port_names = {p.name for p in self.all_ports}
        for port_name in all_component_ports:
            if port_name not in existing_port_names:
                new_port = Port(port_name, len(self.all_ports))
                self.all_ports.append(new_port)
                if verbose: print(f"Created new port: {new_port}")
        
        # Find indices of matching port names
        port_names = [p.name for p in self.all_ports]
        input_indices = [i for i, name in enumerate(port_names) if name in component.input_names]
        output_indices = [i for i, name in enumerate(port_names) if name in component.output_names]
        
        # print("input port indices:", input_indices)
        # print("output port indices:", output_indices)
        # print(f"Total ports: {len(self.all_ports)}")

        indices_1 = [p.index for p in self.all_ports]
        if verbose:
            print("current port indices", indices_1)

        transfer_matrix = np.zeros((len(indices_0), len(indices_1)), dtype=np.complex_)
        if verbose:
            print(np.shape(transfer_matrix.T), "transfer_matrix shape")

        # Set diagonal to 1 for existing ports (identity behavior by default)
        # Only set diagonal for ports that exist in both indices_0 and indices_1
        for i in range(min(len(indices_0), len(indices_1))):
            transfer_matrix[i, i] = 1.0
        
        # Zero out rows for input ports that are involved in the component
        for input_idx in input_indices:
            if input_idx < len(indices_0):  # Ensure input_idx is valid for rows
                transfer_matrix[input_idx, :] = 0  # Zero out the entire row for this input
        
        # Fill the transfer matrix with the component's transfer matrix
        # Map component inputs to device inputs, and component outputs to device outputs
        for i, input_idx in enumerate(input_indices):
            if input_idx < len(indices_0):  # Ensure input_idx is valid for rows
                for j, output_idx in enumerate(output_indices):
                    # Ensure we don't exceed the component's transfer matrix bounds
                    if i < component.transfer_matrix.shape[0] and j < component.transfer_matrix.shape[1]:
                        transfer_matrix[input_idx, output_idx] = component.transfer_matrix[i, j]

        if verbose:
            print("transfer_matrix", transfer_matrix)
        # plt.imshow(np.real(transfer_matrix).T, cmap='gray', vmin=-1, vmax=1)
        # plt.show()
        self.global_matrices.append(transfer_matrix.T)
        
        # Update the device's own transfer matrix to reflect current state
        self.transfer_matrix = self.get_total_transfer_matrix()
        
        # Update output port names to match current all_ports
        self.output_names = [p.name for p in self.all_ports]

    def remove_port(self, port_name):
        """
        Remove a port from the device and update all transfer matrices accordingly.
        
        Args:
            port_name (str): Name of the port to remove
        """
        # Find the port to remove
        port_to_remove = None
        port_index = None
        
        for i, port in enumerate(self.all_ports):
            if port.name == port_name:
                port_to_remove = port
                port_index = i
                break
        
        if port_to_remove is None:
            print(f"Warning: Port '{port_name}' not found in device")
            return
        
        # print(f"Removing port '{port_name}' at index {port_index}")
        
        # Remove the port from all_ports
        self.all_ports.pop(port_index)
        
        # Update indices of remaining ports
        for i, port in enumerate(self.all_ports):
            port.index = i
        
        # Update global matrices to remove the corresponding rows and columns
        updated_matrices = []
        for matrix in self.global_matrices:
            # Remove the row and column corresponding to the removed port
            # matrix shape is (n_ports, n_ports_previous_step)
            # We need to remove the row corresponding to the removed port
            if port_index < matrix.shape[0]:
                # Remove row
                matrix_new = np.delete(matrix, port_index, axis=0)
            else:
                matrix_new = matrix.copy()
            
            updated_matrices.append(matrix_new)
        
        self.global_matrices = updated_matrices
        
        # Update the device's own transfer matrix
        self.transfer_matrix = self.get_total_transfer_matrix()
        
        # Update input/output port names if they contained the removed port
        if port_name in self.input_port_names:
            self.input_port_names.remove(port_name)
            self.input_names.remove(port_name)
        
        if port_name in self.output_names:
            self.output_names.remove(port_name)
        
        # Update output names to match current all_ports
        self.output_names = [p.name for p in self.all_ports]
        
        # print(f"Port '{port_name}' removed. Device now has {len(self.all_ports)} ports: {[p.name for p in self.all_ports]}")
        
    def get_port_names(self):
        """Get list of all current port names"""
        return [p.name for p in self.all_ports]
    
    def get_input_port_names(self):
        """Get list of input port names only"""
        return self.input_port_names.copy()
    
    def remove_ports_by_selection(self, ports_to_keep, component_name=None):
        """
        Remove ports by selecting only the ports to keep using a PortSelector component.
        This will actually reduce the matrix dimensions and remove ports from all_ports.
        
        Args:
            ports_to_keep (list): List of port names to keep
            component_name (str): Optional name for the PortSelector component
        """
        if component_name is None:
            component_name = f"port_selector_{len(self.components)}"
        
        current_ports = self.get_port_names()
        
        # Validate that all ports to keep exist
        missing_ports = [port for port in ports_to_keep if port not in current_ports]
        if missing_ports:
            print(f"Warning: These ports don't exist: {missing_ports}")
            ports_to_keep = [port for port in ports_to_keep if port in current_ports]
        
        if not ports_to_keep:
            print("Error: No valid ports to keep")
            return
        
        # print(f"Selecting ports to keep: {ports_to_keep}")
        # print(f"Removing ports: {[port for port in current_ports if port not in ports_to_keep]}")
        
        # Create and add a PortSelector component
        port_selector = PortSelector(component_name, current_ports, ports_to_keep)
        self.add_component(port_selector)
        
        # Now physically remove the unwanted ports from all_ports
        ports_to_remove = [port for port in current_ports if port not in ports_to_keep]
        for port_name in ports_to_remove:
            # Find and remove the port
            for i, port in enumerate(self.all_ports):
                if port.name == port_name:
                    self.all_ports.pop(i)
                    break
        
        # Update indices of remaining ports
        for i, port in enumerate(self.all_ports):
            port.index = i
        
        # Extract only the relevant part of the transfer matrix (corresponding to remaining ports)
        current_matrix = self.get_total_transfer_matrix()
        kept_indices = [current_ports.index(port_name) for port_name in ports_to_keep]
        
        # Extract submatrix for kept ports only
        reduced_matrix = current_matrix[np.ix_(kept_indices, kept_indices)]
        
        # Update global matrices to reflect the dimension reduction
        # We need to update the global matrices to have the reduced dimensions
        if self.global_matrices:
            # The last global matrix should reflect the final state
            # Replace it with the reduced matrix
            self.global_matrices = [reduced_matrix.T]  # Store as transposed to match convention
        
        # Update the transfer matrix to the reduced size
        self.transfer_matrix = reduced_matrix
        
        # Update output names to match remaining ports
        self.output_names = [p.name for p in self.all_ports]
        
        # print(f"Matrix reduced from {current_matrix.shape} to {reduced_matrix.shape}")
        # print(f"Remaining ports: {self.get_port_names()}")

    def get_total_transfer_matrix(self):
        if len(self.global_matrices) == 0:
            return np.eye(len(self.all_ports), dtype=np.complex_)

        elif len(self.global_matrices) == 1:
            return self.global_matrices[0]

        else:
            ini_mat = self.global_matrices[0]
            for i in range(1, len(self.global_matrices)):
                ini_mat = np.dot(self.global_matrices[i], ini_mat)  # Reversed order
            return ini_mat

class MZI(PIC):
    """Mach-Zehnder Interferometer as a nested device"""
    
    def __init__(self, name, input_names, output_names, phase_shift=0,
                 DC_custom_matrix1 = None, DC_custom_matrix2 = None):
        # Initialize as a Device with input and output ports
        super().__init__(input_names, output_names, name)
        
        # Build the MZI structure: DC -> PhaseShifter -> DC
        # First DC: splits inputs into two arms
        dc1 = DC(f"{name}_dc1", input_names, input_names, custom_matrix=DC_custom_matrix1)
        self.add_component(dc1)
        
        # Phase shifter on one arm (arm1)
        ps = PhaseShifter(f"{name}_ps", [input_names[0]],  [input_names[0]], phase_shift)
        self.add_component(ps)
        
        # Second DC: combines arms back to outputs
        dc2 = DC(f"{name}_dc2", output_names, output_names, custom_matrix=DC_custom_matrix2)
        self.add_component(dc2)
        
        # Extract the relevant part of the transfer matrix that maps inputs to outputs
        total_matrix = self.get_total_transfer_matrix()
        
        # Find indices for input and output ports in the final matrix
        final_port_names = [p.name for p in self.all_ports]
        input_indices_final = [final_port_names.index(name) for name in input_names]
        output_indices_final = [final_port_names.index(name) for name in output_names]
        
        # Extract the input->output submatrix
        self.transfer_matrix = total_matrix[np.ix_(output_indices_final, input_indices_final)]

class ScalarMZI(PIC):

    """ Scalar MZI (intensity modulator) """

    def __init__(self, name, input_names, output_names, phase_shift = 0):

        super().__init__(input_names, output_names, name)
        
        ysplitter = Ysplitter(f"{name}_ys", input_names, [input_names[0], input_names[0]+'_u'])
        self.add_component(ysplitter)

        ps = PhaseShifter(f"{name}_ps",[input_names[0]], [input_names[0]], phase_shift)
        self.add_component(ps)

        ycombiner = Ycombiner(f"{name}_yc",[input_names[0], input_names[0]+'_u'] , output_names)
        self.add_component(ycombiner)

        # remove the unused port
        self.remove_ports_by_selection([input_names[0]])

        # Extract the relevant part of the transfer matrix that maps inputs to outputs
        total_matrix = self.get_total_transfer_matrix()
        
        # Find indices for input and output ports in the final matrix
        final_port_names = [p.name for p in self.all_ports]
        input_indices_final = [final_port_names.index(name) for name in input_names]
        output_indices_final = [final_port_names.index(name) for name in output_names]
        
        # Extract the input->output submatrix
        self.transfer_matrix = total_matrix[np.ix_(output_indices_final, input_indices_final)]

class ABCD(PIC):

    """ ABCD beam combiner"""

    def __init__(self, name, input_names, output_names):
        super().__init__(input_names, output_names, name)
        
        # inputs: a, c
        # outputs: a, c, b, d

        # first split the inputs into two arms
        ysplitter1 = Ysplitter(f"{name}_ys1", [input_names[0]], [output_names[0], output_names[2]], split_ratio=0.5)
        ysplitter2 = Ysplitter(f"{name}_ys2", [input_names[1]], [output_names[1], output_names[3]], split_ratio=0.5)
        self.add_component(ysplitter1)
        self.add_component(ysplitter2)

        # apply pi/2 shift to one of them
        phaseshifter = PhaseShifter(f"{name}_ps", [input_names[0]], [input_names[0]], phase_shift=np.pi/2)
        self.add_component(phaseshifter)

        # two directional couplers
        dc1 = DC(f"{name}_dc1", [output_names[0], output_names[1]], [output_names[0], output_names[1]])
        dc2 = DC(f"{name}_dc2", [output_names[2], output_names[3]], [output_names[2], output_names[3]])
        self.add_component(dc1)
        self.add_component(dc2)

        # extract total matrix
        total_matrix = self.get_total_transfer_matrix()

        # Find indices for input and output ports in the final matrix
        final_port_names = [p.name for p in self.all_ports]
        input_indices_final = [final_port_names.index(name) for name in input_names]
        output_indices_final = [final_port_names.index(name) for name in output_names]

        # Extract the input->output submatrix
        self.transfer_matrix = total_matrix[np.ix_(output_indices_final, input_indices_final)]


class ActivePIC(PIC):
    """
    PIC device with adjustable parameters that updates transfer matrix on-the-fly.
    
    This is a PIC subclass that rebuilds its internal structure when parameters change,
    while maintaining the same interface as a regular PIC device.
    """
    
    def __init__(self, device_template, parameter_names, initial_parameters=None, 
                 input_port_names=None, output_port_names=None, name="Active PIC"):
        """
        Parameters:
        -----------
        device_template : function
            Function that returns a PIC device given parameters
            Example: lambda phase1=0, phase2=0: create_my_device(phase1, phase2)
        parameter_names : list
            Names of the adjustable parameters
        initial_parameters : dict, optional
            Initial parameter values. If None, all parameters start at 0.0
        input_port_names : list, optional
            Input port names. If None, inferred from template device
        output_port_names : list, optional  
            Output port names. If None, inferred from template device
        name : str
            Name of the active PIC device
        """
        self.device_template = device_template
        self.parameter_names = parameter_names
        
        # Initialize parameter values
        if initial_parameters is None:
            self.parameters = {name: 0.0 for name in parameter_names}
        else:
            self.parameters = {name: 0.0 for name in parameter_names}
            self.parameters.update(initial_parameters)
        
        # Cache for built devices to avoid rebuilding identical ones
        self._device_cache = {}
        self._current_built_device = None
        
        # Build initial device to get port structure
        self._rebuild_device()
        
        # If port names not specified, infer from the template device
        if input_port_names is None:
            input_port_names = self._current_built_device.input_port_names
        if output_port_names is None:
            output_port_names = self._current_built_device.get_port_names()
        
        # Initialize as PIC
        super().__init__(input_port_names, output_port_names, name)
        
        # Copy the structure from the built device
        self._copy_structure_from_device()
    
    def _rebuild_device(self):
        """Rebuild the internal device with current parameters."""
        # Create cache key from current parameters
        cache_key = tuple(self.parameters[key] for key in sorted(self.parameters.keys()))
        
        # Check cache first
        if cache_key in self._device_cache:
            self._current_built_device = self._device_cache[cache_key]
        else:
            # Build new device with current parameters
            self._current_built_device = self.device_template(**self.parameters)
            
            # Cache the device
            self._device_cache[cache_key] = self._current_built_device
    
    def _copy_structure_from_device(self):
        """Copy the internal structure from the built device to this PIC."""
        built_device = self._current_built_device
        
        # Clear current structure
        self.components = []
        self.global_matrices = []
        
        # Copy all ports
        self.all_ports = [Port(p.name, p.index) for p in built_device.all_ports]
        
        # Copy components
        self.components = built_device.components.copy()
        
        # Copy global matrices
        self.global_matrices = [mat.copy() for mat in built_device.global_matrices]
        
        # Update transfer matrix
        self.transfer_matrix = built_device.get_total_transfer_matrix()
        
        # Update port names
        self.output_names = [p.name for p in self.all_ports]
    
    def update_parameters(self, **kwargs):
        """
        Update parameter values and rebuild device structure if needed.
        
        Parameters:
        -----------
        **kwargs : dict
            Parameter values to update. Only known parameter names are accepted.
        """
        # Track if any parameters actually changed
        changed = False
        
        # Update parameters
        for key, value in kwargs.items():
            if key in self.parameters:
                if self.parameters[key] != value:
                    self.parameters[key] = value
                    changed = True
            else:
                raise ValueError(f"Unknown parameter: {key}. Known parameters: {list(self.parameters.keys())}")
        
        # Only rebuild if parameters actually changed
        if changed:
            self._rebuild_device()
            self._copy_structure_from_device()
    
    def get_total_transfer_matrix(self, **kwargs):
        """
        Get total transfer matrix with optional parameter updates.
        
        Parameters:
        -----------
        **kwargs : dict
            Optional parameter updates before getting transfer matrix
            
        Returns:
        --------
        transfer_matrix : ndarray
            Current total transfer matrix
        """
        if kwargs:
            self.update_parameters(**kwargs)
        return super().get_total_transfer_matrix()
    
    def get_current_parameters(self):
        """Get current parameter values."""
        return self.parameters.copy()
    
    def get_current_built_device(self):
        """Get the current built PIC device instance."""
        return self._current_built_device
    
    def get_cache_size(self):
        """Get number of cached devices."""
        return len(self._device_cache)
    
    def clear_cache(self):
        """Clear the device cache."""
        self._device_cache = {}
    
    def get_device_info(self):
        """Get information about the active PIC device."""
        info = f"Active PIC: {self.name}\n"
        info += f"Parameters: {self.parameters}\n"
        info += f"Cache size: {len(self._device_cache)} devices\n"
        info += f"Port names: {self.get_port_names()}\n"
        info += f"Transfer matrix shape: {self.get_total_transfer_matrix().shape}\n"
        
        return info
    
    def __repr__(self):
        return self.get_device_info()
