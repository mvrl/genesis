import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class HierarchyIndex:
    """
    Efficient hierarchy index optimized for PyTorch data loading
    All lookups are by location string (e.g., '10_0_218')
    """
    
    def __init__(self):
        # Core mappings (always by location)
        self.location_to_indices = {}  # location -> list of dataset indices
        self.location_to_parent = {}   # location -> parent location (1-level)
        self.location_to_children = defaultdict(list)  # location -> list of child locations
        
        # Multi-level relationships (for efficiency)
        self.location_to_ancestors = defaultdict(dict)  # location -> {zoom_diff: [ancestors]}
        self.location_to_descendants = defaultdict(dict)  # location -> {zoom_diff: [descendants]}
        
        # Metadata
        self.all_locations = set()
        self.isolated_locations = set()  # locations with no parents or children
        
    def add_image_index(self, location, idx):
        """Add dataset index for a location"""
        if location not in self.location_to_indices:
            self.location_to_indices[location] = []
        self.location_to_indices[location].append(idx)
        self.all_locations.add(location)
    
    def add_parent_child(self, parent_loc, child_loc):
        """Add single-level parent-child relationship"""
        self.location_to_parent[child_loc] = parent_loc
        self.location_to_children[parent_loc].append(child_loc)
        self.all_locations.add(parent_loc)
        self.all_locations.add(child_loc)
    
    def finalize(self):
        """Call after adding all relationships to compute derived data"""
        print("Computing multi-level ancestors and descendants...")
        
        # Find isolated locations
        locations_with_relations = (
            set(self.location_to_parent.keys()) | 
            set(self.location_to_children.keys())
        )
        self.isolated_locations = self.all_locations - locations_with_relations
        
        # Compute multi-level ancestors for each location
        for location in tqdm(self.location_to_parent.keys(), desc="Computing ancestors"):
            self._compute_ancestors(location)
        
        # Compute multi-level descendants
        for location in tqdm(self.location_to_children.keys(), desc="Computing descendants"):
            self._compute_descendants(location)
        
        print(f"Finalized: {len(self.all_locations):,} total locations")
        print(f"  - {len(self.isolated_locations):,} isolated")
        print(f"  - {len(self.location_to_parent):,} with parents")
        print(f"  - {len(self.location_to_children):,} with children")

    def set_splits(self, train, random_test, spatial_test, metadata):
        """Attach dataset splits to the hierarchy"""
        self.splits = {
            'train': train,
            'random_test': random_test,
            'spatial_test': spatial_test,
            'metadata': metadata
        }
    
    def _compute_ancestors(self, location, max_levels=5):
        """Compute all ancestors at different zoom levels"""
        current = location
        level = 0
        
        while current in self.location_to_parent and level < max_levels:
            level += 1
            parent = self.location_to_parent[current]
            
            # Calculate zoom difference
            child_zoom = int(location.split('_')[0])
            parent_zoom = int(parent.split('_')[0])
            zoom_diff = child_zoom - parent_zoom
            
            if zoom_diff not in self.location_to_ancestors[location]:
                self.location_to_ancestors[location][zoom_diff] = []
            self.location_to_ancestors[location][zoom_diff].append(parent)
            
            current = parent
    
    def _compute_descendants(self, location, max_levels=5):
        """Compute all descendants at different zoom levels"""
        # BFS to find descendants at each level
        queue = [(location, 0)]
        visited = {location}
        
        while queue:
            current, level = queue.pop(0)
            
            if level >= max_levels:
                continue
            
            for child in self.location_to_children.get(current, []):
                if child in visited:
                    continue
                visited.add(child)
                
                # Calculate zoom difference
                parent_zoom = int(location.split('_')[0])
                child_zoom = int(child.split('_')[0])
                zoom_diff = child_zoom - parent_zoom
                
                if zoom_diff not in self.location_to_descendants[location]:
                    self.location_to_descendants[location][zoom_diff] = []
                self.location_to_descendants[location][zoom_diff].append(child)
                
                queue.append((child, level + 1))
    
    def get_indices(self, location):
        """Get dataset indices for a location"""
        return self.location_to_indices.get(location, [])
    
    def get_parent(self, location):
        """Get immediate parent (1-level up)"""
        return self.location_to_parent.get(location, None)
    
    def get_children(self, location):
        """Get immediate children (1-level down)"""
        return self.location_to_children.get(location, [])
    
    def get_ancestors(self, location, zoom_diff=None):
        """Get ancestors at specific zoom difference or all ancestors"""
        if zoom_diff is not None:
            return self.location_to_ancestors.get(location, {}).get(zoom_diff, [])
        return self.location_to_ancestors.get(location, {})
    
    def get_descendants(self, location, zoom_diff=None):
        """Get descendants at specific zoom difference or all descendants"""
        if zoom_diff is not None:
            return self.location_to_descendants.get(location, {}).get(zoom_diff, [])
        return self.location_to_descendants.get(location, {})
    
    def is_isolated(self, location):
        """Check if location has no parents or children"""
        return location in self.isolated_locations
    
    def find_common_ancestor(self, loc1, loc2):
        """Find lowest common ancestor of two locations"""
        # Get all ancestors of loc1
        ancestors1 = set()
        current = loc1
        while current in self.location_to_parent:
            current = self.location_to_parent[current]
            ancestors1.add(current)
        
        # Walk up from loc2 until we find common ancestor
        current = loc2
        while current in self.location_to_parent:
            current = self.location_to_parent[current]
            if current in ancestors1:
                return current
        
        return None  # No common ancestor
    
    def get_hierarchy_path(self, location):
        """Get full path from location to root"""
        path = [location]
        current = location
        while current in self.location_to_parent:
            current = self.location_to_parent[current]
            path.append(current)
        return path


def build_hierarchy_index(location_to_indices_file, relationships_file, 
                          multi_level_file=None, output_file="hierarchy_index.pkl"):
    """
    Build the consolidated hierarchy index from your existing files
    """
    print("Building consolidated hierarchy index...")
    
    index = HierarchyIndex()
    
    # Load location-to-indices mapping
    print(f"Loading {location_to_indices_file}...")
    with open(location_to_indices_file, 'rb') as f:
        loc_data = pickle.load(f)
        location_to_indices = loc_data['location_to_indices']
    
    # Add all image indices
    print("Adding image indices...")
    for location, indices in tqdm(location_to_indices.items()):
        for idx in indices:
            index.add_image_index(location, idx)
    
    # Load parent-child relationships
    print(f"Loading {relationships_file}...")
    with open(relationships_file, 'rb') as f:
        rel_data = pickle.load(f)
        parent_to_children = rel_data['parent_to_children']
    
    # Add parent-child relationships
    print("Adding parent-child relationships...")
    for parent, children in tqdm(parent_to_children.items()):
        for child in children:
            index.add_parent_child(parent, child)
    
    # Finalize (computes multi-level relationships)
    index.finalize()
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    import os
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved hierarchy index: {output_file} ({size_mb:.1f} MB)")
    
    return index


def load_hierarchy_index(index_file="hierarchy_index.pkl"):
    
    with open(index_file, 'rb') as f:
        index = pickle.load(f)
    
    return index


if __name__ == "__main__":
    # Build once
    # index = build_hierarchy_index(
    #     location_to_indices_file="location_to_indices.pkl",
    #     relationships_file="git10m_relationships.pkl",
    #     output_file="hierarchy_index.pkl"
    # )
    
    # load to test
    index = load_hierarchy_index("hierarchy_index.pkl")
    test_loc = '15_29553_20114'

    print(f"\nTest location: {test_loc}")
    print(f"  Indices: {index.get_indices(test_loc)}")
    print(f"  Parent: {index.get_parent(test_loc)}")
    print(f"  Children: {index.get_children(test_loc)}")
    print(f"  Ancestors (2-level): {index.get_ancestors(test_loc, 2)}")
    print(f"  Isolated: {index.is_isolated(test_loc)}")

    print(index.find_common_ancestor('17_118213_80458', '16_59107_40229'))


# Example usage in PyTorch Dataset
# class SatelliteDataset:
#     def __init__(self, dataset, hierarchy_index):
#         self.dataset = dataset
#         self.hierarchy = hierarchy_index
    
#     def __getitem__(self, idx):
#         # Get the image
#         item = self.dataset['train'][idx]
#         location = item['Google_location']
#         image = item['image']
        
#         # Get hierarchy information efficiently
#         parent = self.hierarchy.get_parent(location)
#         children = self.hierarchy.get_children(location)
#         ancestors_2 = self.hierarchy.get_ancestors(location, zoom_diff=2)
#         ancestor = self.hierarchy.find_common_ancestor(location, parent)
        
#         return {
#             'image': image,
#             'location': location,
#             'parent': parent,
#             'children': children,
#             'ancestors_2': ancestors_2,
#             'is_isolated': self.hierarchy.is_isolated(location)
        # }