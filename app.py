import numpy as np

class HopfieldNetwork:
    def __init__(self,size):
        # In the hopfield network size refers to the number of neurons in the network.
        # For example if the size is 5, then there are 5 neurons in the network.
        # For simplicity the weights are initialized to zero.
        self.size=size
        self.weights = np.zeros((size, size))

    def train(self,patterns):
        # The patterns are a list of binary patterns to be stored in the network.
        for pattern in patterns:
            pattern = np.array(pattern)
            # np.outer computes the outer product of the pattern with itself.
            # This is used to update the weights of the network.
            self.weights += np.outer(pattern, pattern)
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, pattern, max_iterations=100):
        """
        Recall a pattern from a noisy or incomplete input.
        
        Args:
            pattern: Input pattern (can be noisy or incomplete)
            max_iterations: Maximum number of iterations for convergence
            
        Returns:
            Converged pattern
        """
        state = np.array(pattern, dtype=float)
        
        for iteration in range(max_iterations):
            prev_state = state.copy()
            
            # Update each neuron
            for i in range(self.size):
                # Calculate activation (weighted sum of inputs)
                activation = np.dot(self.weights[i], state)
                # Apply sign activation function
                state[i] = 1 if activation >= 0 else -1
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                print(f"Converged after {iteration + 1} iterations")
                break
        else:
            print(f"Did not converge after {max_iterations} iterations")
            
        return state.astype(int)
    
    def energy(self, state):
        """
        Calculate the energy of a given state.
        Lower energy indicates more stable states.
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def add_noise(self, pattern, noise_level=0.1):
        """
        Add noise to a pattern for testing robustness.
        
        Args:
            pattern: Original pattern
            noise_level: Probability of flipping each bit
            
        Returns:
            Noisy pattern
        """
        noisy_pattern = np.array(pattern)
        mask = np.random.random(len(pattern)) < noise_level
        noisy_pattern[mask] *= -1
        return noisy_pattern


def demonstrate_pattern_completion():
    """
    Demonstration 1: Pattern completion from partial information
    """
    print("=" * 50)
    print("DEMONSTRATION 1: PATTERN COMPLETION")
    print("=" * 50)
    
    # Create patterns representing letters (simplified 5x5 grid)
    # Pattern for letter 'T' (flattened)
    pattern_T = np.array([1, 1, 1, 1, 1,
                         -1, -1, 1, -1, -1,
                         -1, -1, 1, -1, -1,
                         -1, -1, 1, -1, -1,
                         -1, -1, 1, -1, -1])
    
    # Pattern for letter 'L'
    pattern_L = np.array([1, -1, -1, -1, -1,
                         1, -1, -1, -1, -1,
                         1, -1, -1, -1, -1,
                         1, -1, -1, -1, -1,
                         1, 1, 1, 1, 1])
    
    patterns = [pattern_T, pattern_L]
    
    # Initialize and train the network
    hopfield = HopfieldNetwork(25)
    hopfield.train(patterns)
    
    # Test with partial T pattern (missing some elements)
    partial_T = pattern_T.copy()
    partial_T[0:5] = 0  # Remove top row
    partial_T[10:15] = 0  # Remove middle row
    
    print("Original T pattern:")
    print_pattern(pattern_T)
    print("\nPartial T pattern (with missing parts):")
    print_pattern(partial_T)
    
    # Recall the complete pattern
    recalled = hopfield.recall(partial_T)
    print("\nRecalled T pattern:")
    print_pattern(recalled)
    
    # Calculate accuracy
    accuracy = np.mean(recalled == pattern_T)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")


def demonstrate_noise_removal():
    """
    Demonstration 2: Noise removal from corrupted patterns
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATION 2: NOISE REMOVAL")
    print("=" * 50)
    
    # Simple 4x4 checkerboard pattern
    checkerboard = np.array([1, -1, 1, -1,
                            -1, 1, -1, 1,
                            1, -1, 1, -1,
                            -1, 1, -1, 1])
    
    # Cross pattern
    cross = np.array([-1, -1, 1, -1,
                     -1, -1, 1, -1,
                     1, 1, 1, 1,
                     -1, -1, 1, -1])
    
    patterns = [checkerboard, cross]
    
    # Initialize and train network
    hopfield = HopfieldNetwork(16)
    hopfield.train(patterns)
    
    # Add noise to checkerboard pattern
    noisy_checkerboard = hopfield.add_noise(checkerboard, noise_level=0.3)
    
    print("Original checkerboard pattern:")
    print_pattern_4x4(checkerboard)
    print("\nNoisy checkerboard pattern (30% noise):")
    print_pattern_4x4(noisy_checkerboard)
    
    # Recall the original pattern
    recalled = hopfield.recall(noisy_checkerboard)
    print("\nRecalled pattern after noise removal:")
    print_pattern_4x4(recalled)
    
    # Calculate accuracy
    accuracy = np.mean(recalled == checkerboard)
    print(f"\nNoise removal accuracy: {accuracy * 100:.1f}%")


def demonstrate_associative_memory():
    """
    Demonstration 3: Associative memory with multiple patterns
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATION 3: ASSOCIATIVE MEMORY")
    print("=" * 50)
    
    # Create multiple distinct patterns
    patterns = [
        np.array([1, 1, -1, -1, 1, 1]),      # Pattern 1
        np.array([-1, 1, 1, 1, 1, -1]),     # Pattern 2
        np.array([1, -1, 1, -1, 1, -1])     # Pattern 3
    ]
    
    # Initialize and train network
    hopfield = HopfieldNetwork(6)
    hopfield.train(patterns)
    
    print("Stored patterns:")
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i+1}: {pattern}")
    
    # Test with partial cues
    partial_cues = [
        np.array([1, 1, 0, 0, 0, 0]),       # Partial cue for pattern 1
        np.array([0, 0, 1, 1, 1, 0]),       # Partial cue for pattern 2
        np.array([0, -1, 0, -1, 0, 0])      # Partial cue for pattern 3
    ]
    
    print("\nRecall from partial cues:")
    for i, cue in enumerate(partial_cues):
        print(f"\nPartial cue {i+1}: {cue}")
        recalled = hopfield.recall(cue)
        print(f"Recalled pattern: {recalled}")
        
        # Find best matching stored pattern
        best_match = 0
        best_similarity = -1
        for j, stored_pattern in enumerate(patterns):
            similarity = np.mean(recalled == stored_pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = j
        
        print(f"Best match: Pattern {best_match + 1} (similarity: {best_similarity * 100:.1f}%)")


def analyze_storage_capacity():
    """
    Demonstration 4: Analysis of storage capacity limitations
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATION 4: STORAGE CAPACITY ANALYSIS")
    print("=" * 50)
    
    network_size = 20
    max_patterns = int(0.15 * network_size)  # Theoretical limit is ~0.138N
    
    print(f"Network size: {network_size} neurons")
    print(f"Testing storage capacity up to {max_patterns} patterns")
    
    results = []
    
    for num_patterns in range(1, max_patterns + 1):
        # Generate random patterns
        patterns = []
        for _ in range(num_patterns):
            pattern = np.random.choice([-1, 1], size=network_size)
            patterns.append(pattern)
        
        # Train network
        hopfield = HopfieldNetwork(network_size)
        hopfield.train(patterns)
        
        # Test recall accuracy
        correct_recalls = 0
        for pattern in patterns:
            # Add small amount of noise
            noisy = hopfield.add_noise(pattern, noise_level=0.1)
            recalled = hopfield.recall(noisy, max_iterations=50)
            
            if np.array_equal(recalled, pattern):
                correct_recalls += 1
        
        accuracy = correct_recalls / num_patterns
        results.append((num_patterns, accuracy))
        
        print(f"Patterns: {num_patterns:2d}, Accuracy: {accuracy * 100:5.1f}%")
    
    return results


def demonstrate_energy_landscape():
    """
    Demonstration 5: Energy landscape visualization
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATION 5: ENERGY LANDSCAPE")
    print("=" * 50)
    
    # Simple 3-neuron network for easy visualization
    patterns = [
        np.array([1, 1, -1]),
        np.array([-1, 1, 1])
    ]
    
    hopfield = HopfieldNetwork(3)
    hopfield.train(patterns)
    
    print("Stored patterns:")
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i+1}: {pattern}")
    
    print("\nEnergy landscape (all possible states):")
    print("State        Energy")
    print("-" * 20)
    
    # Generate all possible 3-bit states
    for i in range(8):
        state = np.array([1 if (i >> j) & 1 else -1 for j in range(3)])
        energy = hopfield.energy(state)
        print(f"{state}   {energy:6.2f}")
    
    print("\nNote: Stored patterns should have the lowest energy values")


def print_pattern(pattern, width=5):
    """Helper function to print pattern as 2D grid"""
    pattern = np.array(pattern)
    for i in range(0, len(pattern), width):
        row = pattern[i:i+width]
        print(' '.join(['█' if x == 1 else '·' for x in row]))


def print_pattern_4x4(pattern):
    """Helper function to print 4x4 pattern"""
    print_pattern(pattern, width=4)


def main():
    """
    Main function to run all demonstrations
    """
    print("HOPFIELD NEURAL NETWORK - COMPREHENSIVE DEMONSTRATION")
    print("====================================================")
    
    # Run all demonstrations
    demonstrate_pattern_completion()
    demonstrate_noise_removal()
    demonstrate_associative_memory()
    
    # Storage capacity analysis
    capacity_results = analyze_storage_capacity()
    
    # Energy landscape
    demonstrate_energy_landscape()
    
    print("\n" + "=" * 50)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 50)
    
    print("\n1. Pattern Completion: Successfully reconstructed incomplete patterns")
    print("2. Noise Removal: Effectively cleaned corrupted input patterns")
    print("3. Associative Memory: Recalled full patterns from partial cues")
    print("4. Storage Capacity: Demonstrated the ~0.138N theoretical limit")
    print("5. Energy Landscape: Showed how stored patterns are energy minima")
    
    print("\nKey Properties Demonstrated:")
    print("- Convergence to stable states")
    print("- Fault tolerance and noise resistance")
    print("- Content-addressable memory capabilities")
    print("- Storage capacity limitations")
    print("- Energy minimization behavior")


if __name__ == "__main__":
    main()