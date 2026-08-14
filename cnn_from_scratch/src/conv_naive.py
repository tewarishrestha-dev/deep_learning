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

        # He initialization
        self.W = np.random.randn(
            num_filters,
            kernel_size,
            kernel_size,
            input_channels
        ) * np.sqrt(
            2 / (kernel_size * kernel_size * input_channels)
        )

        self.b = np.zeros(num_filters)


    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, X):

        self.X = X

        batch_size, height, width, channels = X.shape

        # Padding
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

        # Output dimensions
        output_height = (
            (height - self.kernel_size + 2 * self.padding)
            // self.stride
        ) + 1

        output_width = (
            (width - self.kernel_size + 2 * self.padding)
            // self.stride
        ) + 1

        output = np.zeros(
            (
                batch_size,
                output_height,
                output_width,
                self.num_filters
            )
        )

        # Convolution
        for n in range(batch_size):

            for i in range(output_height):

                for j in range(output_width):

                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size

                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    region = X_padded[
                        n,
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ]

                    for f in range(self.num_filters):

                        output[n, i, j, f] = (
                            np.sum(
                                region * self.W[f]
                            )
                            + self.b[f]
                        )

        return output


    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    def backward(self, d_output):

        batch_size, height, width, channels = self.X.shape

        # Initialize gradients
        dX_padded = np.zeros_like(self.X_padded)

        dW = np.zeros_like(self.W)

        db = np.zeros_like(self.b)


        # Output dimensions
        output_height = d_output.shape[1]
        output_width = d_output.shape[2]


        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        for n in range(batch_size):

            for i in range(output_height):

                for j in range(output_width):

                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size

                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size


                    region = self.X_padded[
                        n,
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ]


                    for f in range(self.num_filters):

                        # Gradient coming from next layer
                        gradient = d_output[
                            n, i, j, f
                        ]


                        # ----------------------------------
                        # Gradient w.r.t. filter weights
                        # ----------------------------------

                        dW[f] += (
                            region * gradient
                        )


                        # ----------------------------------
                        # Gradient w.r.t. bias
                        # ----------------------------------

                        db[f] += gradient


                        # ----------------------------------
                        # Gradient w.r.t. input
                        # ----------------------------------

                        dX_padded[
                            n,
                            h_start:h_end,
                            w_start:w_end,
                            :
                        ] += (
                            self.W[f] * gradient
                        )


        # Remove padding from dX
        if self.padding > 0:

            dX = dX_padded[
                :,
                self.padding:-self.padding,
                self.padding:-self.padding,
                :
            ]

        else:

            dX = dX_padded


        # Store gradients
        self.dW = dW
        self.db = db

        return dX