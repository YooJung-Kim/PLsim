import numpy as np

class Device:

    def __init__(self, 
                 lantern_matrix, 
                 pic_device = None,
                 port_mapping = None,
                 verbose = True
                 ):
        
        self.verbose = verbose
        
        self._validate_inputs(lantern_matrix, pic_device, port_mapping)

    def _validate_inputs(self, lantern_matrix, pic_device, port_mapping):

        if pic_device is not None:
            self.pic = pic_device
            self._has_pic = True
            self.port_mapping = port_mapping
            
            pic_matrix = self._ensure_3d(pic_device.get_total_transfer_matrix())

            n_pic_in = pic_matrix.shape[2]
            n_lan_out = lantern_matrix.shape[1]
            P = np.zeros((n_pic_in, n_lan_out))
            for lan_idx, pic_idx in port_mapping:
                P[pic_idx, lan_idx] = 1.0
            self.matrix = pic_matrix @ P @ self._ensure_3d(lantern_matrix) 
            self.lantern_matrix = lantern_matrix
            self.port_mapping_matrix = P
        else:
            pic_matrix = None
            self._has_pic = False
            self.lantern_matrix = lantern_matrix
            self.matrix = self._ensure_3d(lantern_matrix)

        ndim_lantern_matrix = np.array(lantern_matrix).shape
        self.len_wavelengths = ndim_lantern_matrix[0]

        # self.lantern_matrices = lantern_matrix
        self.pic_matrices = pic_matrix

        if self.verbose:
            print("Device initialized with:")
            print(f"  Lantern matrices: {self.lantern_matrix.shape}")
            print(f"  PIC matrices: {self.pic_matrices.shape if self.pic_matrices is not None else None}")
            print(f"  Wavelengths: {self.len_wavelengths}")

        return

    def _ensure_3d(self, matrix):
        """Enforces (N_wav, Output, Input) shape."""
        arr = np.array(matrix)
        if arr.ndim == 2:
            # User provided single matrix, promote to (1, Out, In)
            return arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            return arr
        else:
            raise ValueError(f"The matrix must be 2D or 3D (found {arr.ndim}D).")

    def calculate_outputs(self, projections):

        # lantern matrices : (nwav x nport x nport)
        # projections: (nwav x nport x nport)

        # spectra = []
        # for wavind in range(self.len_wavelengths):
        #     lantern_matrix = self.lantern_matrices[wavind]
        #     projection = projections[wavind]

        #     total_matrix = lantern_matrix

        #     output = total_matrix @ projection @ np.conj(total_matrix.T)
        #     spectra.append(output)
        # return np.array(spectra)
    

        C = np.array(projections)
        T = self.matrix.copy() # (N_wav, N_out, N_modes)

        if C.shape[0] != self.len_wavelengths:
            if C.shape[0] == self.matrix.shape[2]:
                C = C[np.newaxis, :, :]
            else:
                raise ValueError(f"Input shape {C.shape} incompatible with "
                                  f"N_wav={self.len_wavelengths} or N_modes={T.shape[2]}")
        intensities = np.einsum('wkm, wmn..., wkn -> wk...', T, C, T.conj())
        
        return np.real(intensities)
    

    def update_pic_matrix(self, **kwargs):

        if self._has_pic:
            # self.pic.update_parameters(**kwargs)
            self.pic_matrices = self._ensure_3d(self.pic.get_total_transfer_matrix(**kwargs))
            self.matrix = self.pic_matrices @ self.port_mapping_matrix @ self.lantern_matrix
            if self.verbose:
                    print("PIC parameters updated and matrices recalculated.")
        # else:
        #     raise ValueError("No PIC device to update.")
    