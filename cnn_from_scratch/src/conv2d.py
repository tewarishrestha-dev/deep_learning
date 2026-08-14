import numpy as np


class Conv2D:

    def __init__(
        self,
        input_channels,
        num_filters,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = (
            np.random.randn(
                num_filters,
                kernel_size,
                kernel_size,
                input_channels
            )
            * np.sqrt(
                2.0 / (kernel_size * kernel_size * input_channels)
            )
        )

        self.b = np.zeros(num_filters)

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, X):

        self.X = X

        batch_size, height, width, channels = X.shape

        if self.padding > 0:
            X_padded = np.pad(
                X,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0)
                ),
                mode="constant"
            )
        else:
            X_padded = X

        self.X_padded = X_padded

        output_height = (
            (height - self.kernel_size + 2 * self.padding)
            // self.stride
        ) + 1

        output_width = (
            (width - self.kernel_size + 2 * self.padding)
            // self.stride
        ) + 1

        # Extract all convolution windows
        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            (
                self.kernel_size,
                self.kernel_size
            ),
            axis=(1, 2)
        )

        # sliding_window_view gives:
        # (batch, out_h, out_w, channels, kh, kw)
        #
        # Convert to:
        # (batch, out_h, out_w, kh, kw, channels)

        windows = windows.transpose(
            0, 1, 2, 4, 5, 3
        )

        # Apply stride
        windows = windows[
            :,
            ::self.stride,
            ::self.stride,
            :,
            :,
            :
        ]

        self.windows = windows

        # Flatten receptive fields
        self.X_col = windows.reshape(
            batch_size,
            output_height * output_width,
            -1
        )

        # Flatten filters
        W_col = self.W.reshape(
            self.num_filters,
            -1
        )

        # Matrix multiplication
        output = self.X_col @ W_col.T

        output += self.b

        return output.reshape(
            batch_size,
            output_height,
            output_width,
            self.num_filters
        )

    # ========================================================
    # Backward
    # ========================================================

    def backward(self, d_output):

        batch_size = d_output.shape[0]
        output_height = d_output.shape[1]
        output_width = d_output.shape[2]

        # Flatten upstream gradient
        d_output_flat = d_output.reshape(
            batch_size,
            output_height * output_width,
            self.num_filters
        )

        # Flatten weights
        W_col = self.W.reshape(
            self.num_filters,
            -1
        )

        # ----------------------------------------------------
        # dW
        # ----------------------------------------------------

        dW_col = np.einsum(
            "bpf,bpk->fk",
            d_output_flat,
            self.X_col
        )

        self.dW = dW_col.reshape(
            self.W.shape
        )

        # ----------------------------------------------------
        # db
        # ----------------------------------------------------

        self.db = np.sum(
            d_output_flat,
            axis=(0, 1)
        )

        # ----------------------------------------------------
        # dX patches
        # ----------------------------------------------------

        dX_col = np.einsum(
            "bpf,fk->bpk",
            d_output_flat,
            W_col
        )

        dX_col = dX_col.reshape(
            batch_size,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.input_channels
        )

        # ----------------------------------------------------
        # Scatter patches back into input
        # ----------------------------------------------------

        dX_padded = np.zeros_like(
            self.X_padded
        )

        for i in range(output_height):

            for j in range(output_width):

                h_start = i * self.stride
                w_start = j * self.stride

                dX_padded[
                    :,
                    h_start:h_start + self.kernel_size,
                    w_start:w_start + self.kernel_size,
                    :
                ] += dX_col[:, i, j]

        # Remove padding
        if self.padding > 0:

            dX = dX_padded[
                :,
                self.padding:-self.padding,
                self.padding:-self.padding,
                :
            ]

        else:
            dX = dX_padded

        return dX