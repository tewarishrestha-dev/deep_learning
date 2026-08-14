import numpy as np


class MaxPool2D:

    def __init__(self, pool_size=2, stride=2):

        self.pool_size = pool_size
        self.stride = stride


    def forward(self, X):

        self.X = X

        batch_size, height, width, channels = X.shape

        output_height = (
            (height - self.pool_size) // self.stride
        ) + 1

        output_width = (
            (width - self.pool_size) // self.stride
        ) + 1

        output = np.zeros(
            (
                batch_size,
                output_height,
                output_width,
                channels
            )
        )

        # Store the position of the maximum
        self.max_indices = np.zeros(
            (
                batch_size,
                output_height,
                output_width,
                channels,
                2
            ),
            dtype=int
        )

        for n in range(batch_size):

            for i in range(output_height):

                for j in range(output_width):

                    h_start = i * self.stride
                    h_end = h_start + self.pool_size

                    w_start = j * self.stride
                    w_end = w_start + self.pool_size

                    region = X[
                        n,
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ]

                    for c in range(channels):

                        channel_region = region[:, :, c]

                        max_position = np.unravel_index(
                            np.argmax(channel_region),
                            channel_region.shape
                        )

                        output[n, i, j, c] = (
                            channel_region[max_position]
                        )

                        self.max_indices[
                            n, i, j, c
                        ] = max_position

        return output


    def backward(self, d_output):

        batch_size, height, width, channels = self.X.shape

        dX = np.zeros_like(self.X)

        output_height = d_output.shape[1]
        output_width = d_output.shape[2]

        for n in range(batch_size):

            for i in range(output_height):

                for j in range(output_width):

                    h_start = i * self.stride
                    w_start = j * self.stride

                    for c in range(channels):

                        max_position = self.max_indices[
                            n, i, j, c
                        ]

                        max_h = h_start + max_position[0]
                        max_w = w_start + max_position[1]

                        dX[
                            n,
                            max_h,
                            max_w,
                            c
                        ] += d_output[n, i, j, c]

        return dX