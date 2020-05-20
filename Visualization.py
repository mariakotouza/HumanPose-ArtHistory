import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def visualize2DData (X, values_to_show, fig, ax, image_path, centroids, colors_scatter, datasets):
    """Visualize data in 2d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 2)
        fig - the figure to plot
        ax - the axis
        image_path - the images paths, of shape (N)
        centroids - the clusters' centroids, of shape (#C)
        colors_scatter - the colors to be used in scatter plot, of shape (N)
        datasets - the datasets names
    Returns:
        None
    """

    cmap = plt.cm.RdYlGn
    line = plt.scatter(X[:, 0], X[:, 1], s=10, cmap=cmap, c=colors_scatter)
    plt.legend(handles=line.legend_elements()[0], labels=datasets)

    # create the annotations box
    image = plt.imread(image_path[0])
    im = OffsetImage(image, zoom=0.1)
    xybox = (50., 50.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))

    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    # add skeleton images - to do
    #ab2 = AnnotationBbox(im, (0, 0), xybox=(-20,-40), xycoords='data',
    #                    boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))

    # add it to the axes and make it invisible
    #ax.add_artist(ab2)
    #ab2.set_visible(False)


    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3,
            color='w', zorder=10)

    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (2,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (2,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        #x3, y3 = ax.transData.transform((x2, y2))

        x2 = point[0]
        y2 = point[1]

        # event example:
        # motion_notify_event: xy=(374, 152) xydata=(5.013503440117894, -16.23161532314907) button=None dblclick=False inaxes=AxesSubplot(0.125,0.11;0.775x0.77)

        return np.sqrt ((x2 - event.xdata)**2 + (y2 - event.ydata)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 0:2], event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2 = X[index, 0]
        y2 = X[index, 1]

        annotatePlot.label = plt.annotate( "Value %d" % values_to_show[index],
            xy = (x2, y2), xytext = (-20, -40), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        ind = index
        # get the figure size
        w, h = fig.get_size_inches() * fig.dpi
        ws = (X[index, 0] > w / 2.) * -1 + (X[index, 0] <= w / 2.)
        hs = (X[index, 1] > h / 2.) * -1 + (X[index, 1] <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0] * ws, xybox[1] * hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (x2, y2)
        # set the image corresponding to that point
        im.set_data(plt.imread(image_path[ind]))

        # add skeleton images - to do
        # make annotation box visible
        #ab2.set_visible(True)
        # place it at the position of the hovered scatter point
        #ab2.xy = (x2, y2)
        # set the image corresponding to that point
        #im.set_data(plt.imread(image_path[ind]))

        fig.canvas.draw()


    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        if line.contains(event)[0]:
            closestIndex = calcClosestDatapoint(X, event)
            annotatePlot(X, closestIndex)
        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)

            # add skeleton images - to do
            #ab2.set_visible(False)


    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion


def visualize3DData (X, fig, ax, image_path, C, colors_scatter, datasets):
    """Visualize data in 3d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
    #fig = plt.figure(figsize = (16,10))
    #ax = fig.add_subplot(111, projection = '3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], depthshade = False, picker = True, c=colors_scatter)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

    # create the annotations box
    image = plt.imread(image_path[0])
    im = OffsetImage(image, zoom=0.1)
    xybox = (50., 50.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate( "Value %d" % index,
            xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        ind = index
        # get the figure size
        w, h = fig.get_size_inches() * fig.dpi
        ws = (X[index, 0] > w / 2.) * -1 + (X[index, 0] <= w / 2.)
        hs = (X[index, 1] > h / 2.) * -1 + (X[index, 1] <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0] * ws, xybox[1] * hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (x2, y2)
        # set the image corresponding to that point
        im.set_data(plt.imread(image_path[ind]))

        fig.canvas.draw()


    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot (X, closestIndex)

    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.legend(handles=scatter.legend_elements()[0], labels=datasets, loc=2)

