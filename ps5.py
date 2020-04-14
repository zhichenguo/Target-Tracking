"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.3 * np.eye(4), R=0.5 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """

        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.sigma_t = Q
        self.sigma_dt = Q
        self.sigma_mt = R
        self.Dt = np.array([[1.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
        self.Mt = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0]])
        self.xt_new = None
        self.sigma_t_new = None

    def predict(self):
        self.xt_new = np.dot(self.Dt, self.state)
        self.sigma_t_new = np.dot(np.dot(self.Dt, self.sigma_t), self.Dt.T) + self.sigma_dt

    def correct(self, meas_x, meas_y):
        Kt = np.dot(np.dot(self.sigma_t_new, self.Mt.T), np.linalg.inv(np.dot(np.dot(self.Mt, self.sigma_t_new), self.Mt.T) + self.sigma_mt))
        Yt = np.array([meas_x, meas_y])
        self.state = self.xt_new + np.dot(Kt, (Yt - np.dot(self.Mt, self.xt_new.T).T))
        self.sigma_dt = (np.identity(4) - np.dot(np.dot(Kt, self.Mt), self.sigma_t_new))

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles', 200)  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp', 10.0)  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn', 1)  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        # Initialize any other components you may need when designing your filter.
        self.template = template
        self.frame = frame

        # particles_x = np.random.uniform(self.template.shape[0]/2-1, self.frame.shape[0]-self.template.shape[0]/2+1, size=self.num_particles)
        # particles_y = np.random.uniform(self.template.shape[1]/2-1, self.frame.shape[1]-self.template.shape[1]/2+1, size=self.num_particles)

        x, y, w, h = [self.template_rect[x] for x in ['x', 'y', 'w', 'h']]
        # particles_x = np.random.uniform(x + 2 - h, x - 2 + h, size=self.num_particles)
        # particles_y = np.random.uniform(y + 2 - w, y - 2 + w, size=self.num_particles)

        particles_x = np.random.uniform(x, x + h, size=self.num_particles)
        particles_y = np.random.uniform(y, y + w, size=self.num_particles)

        self.particles = np.column_stack((particles_x, particles_y)).astype(np.float_)

        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        patch = cv2.cvtColor(frame_cutout, cv2.COLOR_BGR2GRAY)
        error = temp - patch
        mse = (error ** 2).mean()

        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, replace=True, p=self.weights)
        new_particles = self.particles[indices]

        return new_particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        for i in range(self.num_particles):

            particle = self.particles[i]

            particle[0] += np.random.normal(0, self.sigma_dyn)
            particle[1] += np.random.normal(0, self.sigma_dyn)

            m = int(self.template.shape[0])
            n = int(self.template.shape[1])
            v = int(particle[0] - m/2)
            u = int(particle[1] - n/2)
            frame_cutout = frame[u:u+m, v:v+n].astype(np.float32)
            if self.template.shape == frame_cutout.shape:
                mse = self.get_error_metric(self.template, frame_cutout)
                self.weights[i] = np.exp(-1.0 * mse / (2 * (self.sigma_exp ** 2)))
            else:
                self.weights[i] = 0

        self.weights = self.weights / self.weights.sum()
        self.particles = self.resample_particles()

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        # x_weighted_mean = 0
        # y_weighted_mean = 0
        #
        # for i in range(self.num_particles):
        #     x_weighted_mean += self.particles[i, 0] * self.weights[i]
        #     y_weighted_mean += self.particles[i, 1] * self.weights[i]
        #
        # x_weighted_mean = x_weighted_mean.astype(np.int)
        # y_weighted_mean = y_weighted_mean.astype(np.int)
        # Complete the rest of the code as instructed.

        # Code comes from Sumeet Anand
        x_mean, y_mean = means = np.average(self.particles, axis=0, weights=self.weights).astype(np.int)

        # draw dots
        for particle in self.particles:
            cv2.circle(frame_in, tuple(particle.astype(int)), radius=1, color=(0, 0, 255), thickness=-1)

        # print self.particles
        # draw rectangle
        h, w = self.template.shape[:2]
        # print x_mean, y_mean,  h, w
        cv2.rectangle(frame_in, (x_mean - w/2, y_mean - h/2), (x_mean + w/2, y_mean + h/2), color=(192, 192, 192), thickness=1)

        # draw circle
        particles_dist = ((self.particles - means) ** 2).sum(axis=1) ** 0.5
        radius = np.average(particles_dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, tuple(means), radius=radius, color=(192, 192, 192), thickness=1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha', 0.3)  # required by the autograder
        self.last_template = template
        # self.num_particles *= 2
        # self.best = template
        # self.num_best = 1
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """


        last_best = np.average(self.particles, axis=0, weights=self.weights)
        # print mean_loc
        # last_best = self.particles[np.argmax(self.weights)]
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        # frame_green = frame[:,:,1]
        # template_green = self.last_template[:,:,1]
        # last_best_t = frame_green[(int(last_best[0]-template_green.shape[1]/2)): (int(last_best[0]-template_green.shape[1]/2+template_green.shape[1])),
        #          (int(last_best[1]-template_green.shape[0]/2)): (int(last_best[1]-template_green.shape[0]/2+template_green.shape[0]))]

        last_best_t = frame[(int(last_best[0]-self.last_template.shape[1]/2)): (int(last_best[0]-self.last_template.shape[1]/2+self.last_template.shape[1])),
                 (int(last_best[1]-self.last_template.shape[0]/2)): (int(last_best[1]-self.last_template.shape[0]/2+self.last_template.shape[0]))]

        if last_best.shape == self.last_template.shape:
             self.last_template = last_best_t
        # self.last_template = last_best_t

        for i in range(self.num_particles):

            particle = self.particles[i]
            particle[0] += np.random.normal(0, self.sigma_dyn)
            particle[1] += np.random.normal(0, self.sigma_dyn)
            # threshold = np.sqrt((self.template.shape[0]/2)**2 + (self.template.shape[1]/2)**2)
            # distance = np.sqrt((last_best[0] - particle[0])**2 + (last_best[1] - particle[1])**2)

            if np.abs(last_best[0] - particle[0]) < self.template.shape[0]/6 and np.abs(last_best[1] - particle[1]) < self.template.shape[1]/6:

                m = int(self.template.shape[0])
                n = int(self.template.shape[1])
                v = int(particle[0] - m/2)
                u = int(particle[1] - n/2)
                frame_cutout = frame[u:u+m, v:v+n].astype(np.float32)

                if self.template.shape == frame_cutout.shape:
                    mse = self.get_error_metric(self.template.astype(np.float32), frame_cutout)
                    self.weights[i] = np.exp(-1.0 * mse / (2 * (self.sigma_exp ** 2)))
                else:
                    self.weights[i] = 0
            else:
                self.weights[i] = 0

        self.weights = self.weights / self.weights.sum()

        self.particles = self.resample_particles()

        # best = self.particles[np.argmax(self.weights)]
        best = np.average(self.particles, axis=0, weights=self.weights)
        best_t = frame[(int(best[0]-self.template.shape[1]/2)): (int(best[0]-self.template.shape[1]/2+self.template.shape[1])),
                 (int(best[1]-self.template.shape[0]/2)): (int(best[1]-self.template.shape[0]/2+self.template.shape[0]))]

        # frame_green = frame[:,:,1]
        # template_green = self.last_template[:,:,1]
        # best_t = frame_green[(int(best[0]-template_green.shape[1]/2)): (int(best[0]-template_green.shape[1]/2+template_green.shape[1])),
        #          (int(best[1]-template_green.shape[0]/2)): (int(best[1]-template_green.shape[0]/2+template_green.shape[0]))]

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        # best_t = frame_gray[(int(best[0]-template_gray.shape[1]/2)): (int(best[0]-template_gray.shape[1]/2+template_gray.shape[1])),
        #          (int(best[1]-template_gray.shape[0]/2)): (int(best[1]-template_gray.shape[0]/2+template_gray.shape[0]))]
        if best_t.shape == self.template.shape:
             self.template = self.alpha * best_t + (1 - self.alpha) * self.last_template



class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.last_template = template
        self.alpha = kwargs.get('alpha', 0.85)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        last_best = np.average(self.particles, axis=0, weights=self.weights)

        last_best_t = frame[(int(last_best[0]-self.last_template.shape[1]/2)): (int(last_best[0]-self.last_template.shape[1]/2+self.last_template.shape[1])),
                      (int(last_best[1]-self.last_template.shape[0]/2)): (int(last_best[1]-self.last_template.shape[0]/2+self.last_template.shape[0]))]

        if last_best.shape == self.last_template.shape:
             self.last_template = last_best_t
        # self.last_template = last_best_t

        for i in range(self.num_particles):

            particle = self.particles[i]
            particle[0] += np.random.normal(0, self.sigma_dyn)
            particle[1] += np.random.normal(0, self.sigma_dyn)
            # threshold = np.sqrt((self.template.shape[0]/2)**2 + (self.template.shape[1]/2)**2)
            # distance = np.sqrt((last_best[0] - particle[0])**2 + (last_best[1] - particle[1])**2)

            # if np.abs(last_best[0] - particle[0]) < self.template.shape[0]/2 and np.abs(last_best[1] - particle[1]) < self.template.shape[1]/2:

            if True:

                m = int(self.template.shape[0])
                n = int(self.template.shape[1])
                v = int(particle[0] - m/2)
                u = int(particle[1] - n/2)
                frame_cutout = frame[u:u+m, v:v+n].astype(np.float32)

                if self.template.shape == frame_cutout.shape:
                    mse = self.get_error_metric(self.template.astype(np.float32), frame_cutout)
                    self.weights[i] = np.exp(-1.0 * mse / (2 * (self.sigma_exp ** 2)))
                else:
                    self.weights[i] = 0
            else:
                self.weights[i] = 0

        self.weights = self.weights / self.weights.sum()

        self.particles = self.resample_particles()

        best = np.average(self.particles, axis=0, weights=self.weights)
        best_t = frame[(int(best[0]-self.template.shape[1]/2)): (int(best[0]-self.template.shape[1]/2+self.template.shape[1])),
                 (int(best[1]-self.template.shape[0]/2)): (int(best[1]-self.template.shape[0]/2+self.template.shape[0]))]

        if best_t.shape == self.last_template.shape:
             self.template = self.alpha * best_t + (1 - self.alpha) * self.last_template

        resized_template = cv2.resize(self.template.copy(), (0, 0), fx=0.98, fy=0.99)
        self.template = resized_template

