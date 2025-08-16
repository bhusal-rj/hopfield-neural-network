import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import param
from PIL import Image
import cv2
from skimage import data, filters, measure
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
import warnings
warnings.filterwarnings('ignore')

# Enable Panel extensions
pn.extension('bokeh', 'matplotlib')

class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize Hopfield Neural Network
        
        Args:
            size: Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size))
        self.training_history = []
        self.recall_history = []
        
    def train(self, patterns):
        """
        Train the network using Hebbian learning rule
        
        Args:
            patterns: List of binary patterns to store
        """
        self.weights = np.zeros((self.size, self.size))
        
        for pattern in patterns:
            pattern = np.array(pattern)
            # Store the outer product
            self.weights += np.outer(pattern, pattern)
            
        # Normalize by number of patterns
        self.weights /= len(patterns)
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Store training info
        self.training_history.append({
            'num_patterns': len(patterns),
            'patterns': [p.copy() for p in patterns],
            'weights_norm': np.linalg.norm(self.weights)
        })
    
    def recall(self, pattern, max_iterations=100, track_energy=False):
        """
        Recall a pattern from noisy/incomplete input
        
        Args:
            pattern: Input pattern
            max_iterations: Maximum iterations for convergence
            track_energy: Whether to track energy during recall
            
        Returns:
            Converged pattern and optional energy history
        """
        state = np.array(pattern, dtype=float)
        energy_history = []
        state_history = [state.copy()]
        
        for iteration in range(max_iterations):
            prev_state = state.copy()
            
            # Calculate current energy if tracking
            if track_energy:
                energy = self.energy(state)
                energy_history.append(energy)
            
            # Update each neuron asynchronously
            for i in range(self.size):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
                
            state_history.append(state.copy())
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                if track_energy:
                    energy_history.append(self.energy(state))
                break
                
        # Store recall info
        self.recall_history.append({
            'iterations': iteration + 1,
            'converged': np.array_equal(state, prev_state),
            'final_energy': self.energy(state),
            'state_history': state_history
        })
        
        if track_energy:
            return state.astype(int), energy_history
        return state.astype(int)
    
    def energy(self, state):
        """Calculate energy of the network state"""
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def add_noise(self, pattern, noise_level=0.1):
        """Add noise to a pattern"""
        noisy_pattern = np.array(pattern)
        mask = np.random.random(len(pattern)) < noise_level
        noisy_pattern[mask] *= -1
        return noisy_pattern


class ImageDenoiser:
    """Hopfield Network for Image Denoising"""
    
    def __init__(self, image_size=(8, 8)):
        """
        Initialize image denoiser
        
        Args:
            image_size: Size of image patches to process
        """
        self.image_size = image_size
        self.patch_size = image_size[0] * image_size[1]
        self.hopfield = HopfieldNetwork(self.patch_size)
        
    def preprocess_image(self, image):
        """Convert image to binary patches"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize image to be divisible by patch size
        h, w = image.shape
        new_h = (h // self.image_size[0]) * self.image_size[0]
        new_w = (w // self.image_size[1]) * self.image_size[1]
        image = cv2.resize(image, (new_w, new_h))
        
        # Convert to binary
        threshold = np.mean(image)
        binary_image = np.where(image > threshold, 1, -1)
        
        return binary_image
    
    def extract_patches(self, image):
        """Extract non-overlapping patches from image"""
        h, w = image.shape
        patches = []
        
        for i in range(0, h, self.image_size[0]):
            for j in range(0, w, self.image_size[1]):
                patch = image[i:i+self.image_size[0], j:j+self.image_size[1]]
                if patch.shape == self.image_size:
                    patches.append(patch.flatten())
                    
        return np.array(patches)
    
    def reconstruct_image(self, patches, original_shape):
        """Reconstruct image from patches"""
        h, w = original_shape
        reconstructed = np.zeros((h, w))
        
        patch_idx = 0
        for i in range(0, h, self.image_size[0]):
            for j in range(0, w, self.image_size[1]):
                if patch_idx < len(patches):
                    patch = patches[patch_idx].reshape(self.image_size)
                    reconstructed[i:i+self.image_size[0], j:j+self.image_size[1]] = patch
                    patch_idx += 1
                    
        return reconstructed
    
    def train_on_image(self, clean_image):
        """Train the network on clean image patches"""
        processed_image = self.preprocess_image(clean_image)
        patches = self.extract_patches(processed_image)
        
        # Select representative patches (avoid too many similar patches)
        unique_patches = []
        for patch in patches:
            is_unique = True
            for existing in unique_patches:
                if np.array_equal(patch, existing):
                    is_unique = False
                    break
            if is_unique and len(unique_patches) < int(0.138 * self.patch_size):
                unique_patches.append(patch)
                
        self.hopfield.train(unique_patches)
        return processed_image, unique_patches
    
    def denoise_image(self, noisy_image):
        """Denoise an image using the trained network"""
        processed_image = self.preprocess_image(noisy_image)
        patches = self.extract_patches(processed_image)
        
        denoised_patches = []
        for patch in patches:
            denoised_patch = self.hopfield.recall(patch, max_iterations=50)
            denoised_patches.append(denoised_patch)
            
        denoised_image = self.reconstruct_image(denoised_patches, processed_image.shape)
        return denoised_image


class HopfieldVisualizer(param.Parameterized):
    """Interactive visualization of Hopfield Network"""
    
    # Parameters for interactive control
    network_size = param.Integer(default=25, bounds=(9, 100))
    noise_level = param.Number(default=0.2, bounds=(0, 0.5))
    pattern_type = param.Selector(default='letters', objects=['letters', 'digits', 'custom'])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.hopfield = HopfieldNetwork(self.network_size)
        self.patterns = self._generate_patterns()
        
    def _generate_patterns(self):
        """Generate patterns based on type"""
        if self.pattern_type == 'letters':
            return self._generate_letter_patterns()
        elif self.pattern_type == 'digits':
            return self._generate_digit_patterns()
        else:
            return self._generate_custom_patterns()
    
    def _generate_letter_patterns(self):
        """Generate letter patterns for current network size"""
        if self.network_size == 25:  # 5x5 grid
            # Letter T
            pattern_T = np.array([
                1, 1, 1, 1, 1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1
            ])
            
            # Letter L
            pattern_L = np.array([
                1, -1, -1, -1, -1,
                1, -1, -1, -1, -1,
                1, -1, -1, -1, -1,
                1, -1, -1, -1, -1,
                1, 1, 1, 1, 1
            ])
            
            # Letter I
            pattern_I = np.array([
                1, 1, 1, 1, 1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                1, 1, 1, 1, 1
            ])
            
            return [pattern_T, pattern_L, pattern_I]
        
        elif self.network_size == 9:  # 3x3 grid
            # Simple 3x3 patterns
            pattern_1 = np.array([
                1, 1, 1,
                -1, 1, -1,
                -1, 1, -1
            ])
            
            pattern_2 = np.array([
                1, -1, -1,
                1, -1, -1,
                1, 1, 1
            ])
            
            return [pattern_1, pattern_2]
        
        else:
            # Generate random patterns for other sizes
            return self._generate_custom_patterns()
    
    def _generate_digit_patterns(self):
        """Generate digit patterns for current network size"""
        if self.network_size == 25:  # 5x5 grid
            # Simple 5x5 digit patterns
            digit_0 = np.array([
                1, 1, 1, 1, 1,
                1, -1, -1, -1, 1,
                1, -1, -1, -1, 1,
                1, -1, -1, -1, 1,
                1, 1, 1, 1, 1
            ])
            
            digit_1 = np.array([
                -1, -1, 1, -1, -1,
                -1, 1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, 1, 1, 1, -1
            ])
            
            return [digit_0, digit_1]
        
        else:
            # Generate simple patterns for other sizes
            return self._generate_custom_patterns()
    
    def _generate_custom_patterns(self):
        """Generate custom random patterns"""
        patterns = []
        for _ in range(3):
            pattern = np.random.choice([-1, 1], size=self.network_size)
            patterns.append(pattern)
        return patterns
    
    def train_network(self):
        """Train the network with current patterns"""
        self.hopfield = HopfieldNetwork(self.network_size)
        self.hopfield.train(self.patterns)
    
    def create_pattern_plot(self):
        """Create pattern visualization"""
        fig, axes = plt.subplots(1, len(self.patterns), figsize=(12, 4))
        if len(self.patterns) == 1:
            axes = [axes]
            
        for i, pattern in enumerate(self.patterns):
            grid_size = int(np.sqrt(len(pattern)))
            pattern_2d = pattern.reshape(grid_size, grid_size)
            
            axes[i].imshow(pattern_2d, cmap='RdBu', vmin=-1, vmax=1)
            axes[i].set_title(f'Pattern {i+1}')
            axes[i].axis('off')
            
        plt.tight_layout()
        return fig
    
    def create_recall_demo(self):
        """Create recall demonstration"""
        self.train_network()
        
        # Add noise to first pattern
        original = self.patterns[0]
        noisy = self.hopfield.add_noise(original, self.noise_level)
        recalled, energy_history = self.hopfield.recall(noisy, track_energy=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        grid_size = int(np.sqrt(len(original)))
        
        # Original pattern
        axes[0,0].imshow(original.reshape(grid_size, grid_size), cmap='RdBu', vmin=-1, vmax=1)
        axes[0,0].set_title('Original Pattern')
        axes[0,0].axis('off')
        
        # Noisy pattern
        axes[0,1].imshow(noisy.reshape(grid_size, grid_size), cmap='RdBu', vmin=-1, vmax=1)
        axes[0,1].set_title(f'Noisy Pattern ({self.noise_level*100:.0f}% noise)')
        axes[0,1].axis('off')
        
        # Recalled pattern
        axes[1,0].imshow(recalled.reshape(grid_size, grid_size), cmap='RdBu', vmin=-1, vmax=1)
        axes[1,0].set_title('Recalled Pattern')
        axes[1,0].axis('off')
        
        # Energy evolution
        axes[1,1].plot(energy_history, 'b-o', markersize=4)
        axes[1,1].set_title('Energy During Recall')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_energy_landscape(self):
        """Create energy landscape visualization for small networks"""
        if self.network_size > 16:
            return None  # Too many states to visualize
            
        self.train_network()
        
        # Generate all possible states
        num_states = 2 ** self.network_size
        states = []
        energies = []
        
        for i in range(num_states):
            state = np.array([1 if (i >> j) & 1 else -1 for j in range(self.network_size)])
            states.append(state)
            energies.append(self.hopfield.energy(state))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(energies))
        bars = ax.bar(x, energies, alpha=0.7)
        
        # Highlight stored patterns
        for i, state in enumerate(states):
            for pattern in self.patterns:
                if np.array_equal(state, pattern):
                    bars[i].set_color('red')
                    bars[i].set_alpha(1.0)
        
        ax.set_title('Energy Landscape (Red bars = Stored Patterns)')
        ax.set_xlabel('State Index')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
        
        return fig


def create_image_denoising_demo():
    """Create image denoising demonstration"""
    print("Creating image denoising demonstration...")
    
    # Create simple test images
    def create_test_image(size=(32, 32)):
        """Create a simple test image with patterns"""
        image = np.zeros(size)
        
        # Add some geometric patterns
        h, w = size
        
        # Horizontal lines
        image[h//4, :] = 255
        image[3*h//4, :] = 255
        
        # Vertical lines
        image[:, w//4] = 255
        image[:, 3*w//4] = 255
        
        # Rectangle
        image[h//2-5:h//2+5, w//2-10:w//2+10] = 255
        
        return image
    
    # Create test image
    clean_image = create_test_image()
    
    # Initialize denoiser
    denoiser = ImageDenoiser(image_size=(8, 8))
    
    # Train on clean image
    processed_clean, training_patches = denoiser.train_on_image(clean_image)
    
    # Add noise
    noise = np.random.normal(0, 50, clean_image.shape)
    noisy_image = clean_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Denoise
    denoised_image = denoiser.denoise_image(noisy_image)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0,0].imshow(clean_image, cmap='gray')
    axes[0,0].set_title('Original Clean Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(noisy_image, cmap='gray')
    axes[0,1].set_title('Noisy Image')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(denoised_image, cmap='RdBu')
    axes[0,2].set_title('Denoised Image')
    axes[0,2].axis('off')
    
    # Processed versions
    axes[1,0].imshow(processed_clean, cmap='RdBu')
    axes[1,0].set_title('Processed Clean (Binary)')
    axes[1,0].axis('off')
    
    # Show some training patches
    patch_grid = np.zeros((24, 24))
    for i, patch in enumerate(training_patches[:9]):
        row, col = divmod(i, 3)
        patch_2d = patch.reshape(8, 8)
        patch_grid[row*8:(row+1)*8, col*8:(col+1)*8] = patch_2d
    
    axes[1,1].imshow(patch_grid, cmap='RdBu')
    axes[1,1].set_title('Training Patches')
    axes[1,1].axis('off')
    
    # Error analysis
    error = np.abs(processed_clean - denoised_image)
    axes[1,2].imshow(error, cmap='Reds')
    axes[1,2].set_title('Reconstruction Error')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    return fig


def create_interactive_dashboard():
    """Create interactive Panel dashboard"""
    
    # Create visualizer
    visualizer = HopfieldVisualizer()
    
    # Create parameter panel
    param_panel = pn.Param(
        visualizer,
        parameters=['network_size', 'noise_level', 'pattern_type'],
        widgets={
            'network_size': pn.widgets.IntSlider,
            'noise_level': pn.widgets.FloatSlider,
            'pattern_type': pn.widgets.Select
        }
    )
    
    # Create plot functions
    @pn.depends(visualizer.param.network_size, visualizer.param.pattern_type)
    def pattern_plot():
        visualizer.patterns = visualizer._generate_patterns()
        return visualizer.create_pattern_plot()
    
    @pn.depends(visualizer.param.network_size, visualizer.param.noise_level, visualizer.param.pattern_type)
    def recall_plot():
        visualizer.patterns = visualizer._generate_patterns()
        return visualizer.create_recall_demo()
    
    @pn.depends(visualizer.param.network_size, visualizer.param.pattern_type)
    def energy_plot():
        visualizer.patterns = visualizer._generate_patterns()
        fig = visualizer.create_energy_landscape()
        if fig is None:
            return pn.pane.HTML("<p>Energy landscape visualization only available for networks with ≤16 neurons</p>")
        return fig
    
    # Create dashboard layout
    dashboard = pn.template.FastListTemplate(
        title="Hopfield Neural Network Interactive Demo",
        sidebar=[param_panel],
        main=[
            pn.Column(
                "## Stored Patterns",
                pattern_plot,
                "## Pattern Recall Demonstration", 
                recall_plot,
                "## Energy Landscape",
                energy_plot
            )
        ]
    )
    
    return dashboard


def main():
    """Main function to run enhanced demonstrations"""
    print("HOPFIELD NEURAL NETWORK - ENHANCED WITH VISUALIZATION")
    print("=" * 60)
    
    # 1. Basic demonstrations
    print("\n1. Running basic pattern demonstrations...")
    visualizer = HopfieldVisualizer(network_size=25)
    
    # Create and show pattern plots
    fig1 = visualizer.create_pattern_plot()
    plt.savefig('patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create and show recall demo
    fig2 = visualizer.create_recall_demo()
    plt.savefig('recall_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Image denoising demonstration
    print("\n2. Running image denoising demonstration...")
    fig3 = create_image_denoising_demo()
    plt.savefig('image_denoising.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Energy landscape for small network
    print("\n3. Creating energy landscape visualization...")
    small_visualizer = HopfieldVisualizer(network_size=9)
    small_visualizer.patterns = small_visualizer._generate_patterns()
    fig4 = small_visualizer.create_energy_landscape()
    if fig4:
        plt.savefig('energy_landscape.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n4. Creating interactive dashboard...")
    print("Run 'panel serve hopfield_enhanced.py --show' to launch interactive dashboard")
    
    # 5. Performance analysis
    print("\n5. Performance Analysis with Visualization...")
    analyze_performance_with_plots()


def analyze_performance_with_plots():
    """Analyze performance with visualization"""
    
    # Test different network sizes and noise levels
    network_sizes = [16, 25, 36, 49, 64]
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    
    for size in network_sizes:
        for noise in noise_levels:
            # Create test patterns
            num_patterns = max(1, int(0.1 * size))  # Conservative pattern count
            patterns = []
            for _ in range(num_patterns):
                pattern = np.random.choice([-1, 1], size=size)
                patterns.append(pattern)
            
            # Train network
            hopfield = HopfieldNetwork(size)
            hopfield.train(patterns)
            
            # Test recall accuracy
            correct_recalls = 0
            total_tests = min(10, len(patterns))
            
            for i in range(total_tests):
                original = patterns[i % len(patterns)]
                noisy = hopfield.add_noise(original, noise)
                recalled = hopfield.recall(noisy, max_iterations=50)
                
                if np.array_equal(recalled, original):
                    correct_recalls += 1
            
            accuracy = correct_recalls / total_tests
            results.append((size, noise, accuracy))
    
    # Create performance plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy vs Network Size
    for noise in noise_levels:
        size_data = [(size, acc) for size, n, acc in results if n == noise]
        sizes, accuracies = zip(*size_data)
        axes[0].plot(sizes, accuracies, 'o-', label=f'Noise {noise:.1f}')
    
    axes[0].set_xlabel('Network Size')
    axes[0].set_ylabel('Recall Accuracy')
    axes[0].set_title('Accuracy vs Network Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Noise Level
    for size in network_sizes:
        noise_data = [(noise, acc) for s, noise, acc in results if s == size]
        noises, accuracies = zip(*noise_data)
        axes[1].plot(noises, accuracies, 'o-', label=f'Size {size}')
    
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Recall Accuracy')
    axes[1].set_title('Accuracy vs Noise Level')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
    
    # For Panel dashboard, uncomment the following:
    # dashboard = create_interactive_dashboard()
    # dashboard.servable()
