"""Unit tests for the model registry module."""

import os
import shutil
import pytest
import numpy as np
from datetime import datetime

from utchs.utils.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a temporary registry for testing."""
    registry = ModelRegistry(base_dir=str(tmp_path))
    yield registry
    # Cleanup
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


def test_registry_initialization(registry, tmp_path):
    """Test registry initialization."""
    assert os.path.exists(tmp_path)
    assert os.path.exists(os.path.join(tmp_path, "versions"))
    assert os.path.exists(os.path.join(tmp_path, "metadata"))
    assert os.path.exists(os.path.join(tmp_path, "index.json"))


def test_model_registration(registry):
    """Test model registration."""
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={"param1": 1.0, "param2": "test"},
        artifacts={"artifact1": "path/to/artifact"}
    )
    
    assert model_id is not None
    model_info = registry.get_model(model_id)
    assert model_info["name"] == "test_model"
    assert model_info["description"] == "Test model"
    assert model_info["parameters"] == {"param1": 1.0, "param2": "test"}
    assert model_info["artifacts"] == {"artifact1": "path/to/artifact"}


def test_version_management(registry):
    """Test version management."""
    # Register model
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={}
    )
    
    # Add version
    version_id = registry.add_version(
        model_id=model_id,
        version_name="v1",
        data={"field": np.zeros((10, 10))},
        metrics={"accuracy": 0.95},
        parent_version=None
    )
    
    assert version_id is not None
    
    # Get version info and data
    version_info, version_data = registry.get_version(model_id, version_id)
    assert version_info["name"] == "v1"
    assert version_info["metrics"] == {"accuracy": 0.95}
    assert "field" in version_data
    assert version_data["field"].shape == (10, 10)


def test_field_management(registry):
    """Test field management."""
    # Register model and version
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={}
    )
    
    version_id = registry.add_version(
        model_id=model_id,
        version_name="v1",
        data={},
        metrics={}
    )
    
    # Create test field
    field = np.random.rand(10, 10)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {"param1": 1.0}
    }
    
    # Save field
    path = registry.save_field(
        model_id=model_id,
        version_id=version_id,
        field=field,
        name="test_field",
        metadata=metadata
    )
    
    assert os.path.exists(path)
    
    # Load field
    loaded_field, loaded_metadata = registry.load_field(
        model_id=model_id,
        version_id=version_id,
        name="test_field"
    )
    
    assert np.array_equal(field, loaded_field)
    assert loaded_metadata["model_id"] == model_id
    assert loaded_metadata["version_id"] == version_id
    assert "timestamp" in loaded_metadata
    assert loaded_metadata["parameters"] == {"param1": 1.0}


def test_version_comparison(registry):
    """Test version comparison."""
    # Register model
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={}
    )
    
    # Add versions
    version_ids = []
    for i in range(3):
        version_id = registry.add_version(
            model_id=model_id,
            version_name=f"v{i+1}",
            data={},
            metrics={"accuracy": 0.9 + i*0.01},
            parent_version=version_ids[-1] if version_ids else None
        )
        version_ids.append(version_id)
    
    # Compare versions
    comparison = registry.compare_versions(
        model_id=model_id,
        version_ids=version_ids,
        metric="accuracy"
    )
    
    assert comparison["metric"] == "accuracy"
    assert len(comparison["comparison"]) == 3
    assert comparison["comparison"][version_ids[0]] == 0.9
    assert comparison["comparison"][version_ids[1]] == 0.91
    assert comparison["comparison"][version_ids[2]] == 0.92


def test_model_export_import(registry, tmp_path):
    """Test model export and import."""
    # Register model and add version
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={"param1": 1.0}
    )
    
    version_id = registry.add_version(
        model_id=model_id,
        version_name="v1",
        data={"field": np.zeros((10, 10))},
        metrics={"accuracy": 0.95}
    )
    
    # Save field
    field = np.random.rand(10, 10)
    registry.save_field(
        model_id=model_id,
        version_id=version_id,
        field=field,
        name="test_field"
    )
    
    # Export model
    export_dir = os.path.join(tmp_path, "export")
    exported_path = registry.export_model(model_id, export_dir)
    
    assert os.path.exists(exported_path)
    assert os.path.exists(os.path.join(exported_path, "model.json"))
    assert os.path.exists(os.path.join(exported_path, "versions"))
    
    # Import model
    new_model_id = registry.import_model(export_dir)
    
    # Verify imported model
    new_model_info = registry.get_model(new_model_id)
    assert new_model_info["name"] == "test_model"
    assert new_model_info["parameters"] == {"param1": 1.0}
    
    # Get imported version
    new_version_id = new_model_info["versions"][0]["id"]
    new_version_info, new_version_data = registry.get_version(new_model_id, new_version_id)
    assert new_version_info["name"] == "v1"
    assert new_version_info["metrics"] == {"accuracy": 0.95}
    
    # Load imported field
    loaded_field, _ = registry.load_field(
        model_id=new_model_id,
        version_id=new_version_id,
        name="test_field"
    )
    assert np.array_equal(field, loaded_field)


def test_model_deletion(registry):
    """Test model deletion."""
    # Register model and add version
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={}
    )
    
    version_id = registry.add_version(
        model_id=model_id,
        version_name="v1",
        data={},
        metrics={}
    )
    
    # Save field
    registry.save_field(
        model_id=model_id,
        version_id=version_id,
        field=np.zeros((10, 10)),
        name="test_field"
    )
    
    # Delete model
    registry.delete_model(model_id)
    
    # Verify deletion
    with pytest.raises(ValueError):
        registry.get_model(model_id)
    
    # Verify version directory is deleted
    version_dir = os.path.join(registry.base_dir, "versions", version_id)
    assert not os.path.exists(version_dir)


def test_version_deletion(registry):
    """Test version deletion."""
    # Register model and add versions
    model_id = registry.register_model(
        name="test_model",
        description="Test model",
        parameters={}
    )
    
    version_ids = []
    for i in range(2):
        version_id = registry.add_version(
            model_id=model_id,
            version_name=f"v{i+1}",
            data={},
            metrics={}
        )
        version_ids.append(version_id)
        
        registry.save_field(
            model_id=model_id,
            version_id=version_id,
            field=np.zeros((10, 10)),
            name="test_field"
        )
    
    # Delete first version
    registry.delete_version(model_id, version_ids[0])
    
    # Verify deletion
    with pytest.raises(ValueError):
        registry.get_version(model_id, version_ids[0])
    
    # Verify version directory is deleted
    version_dir = os.path.join(registry.base_dir, "versions", version_ids[0])
    assert not os.path.exists(version_dir)
    
    # Verify second version still exists
    version_info, _ = registry.get_version(model_id, version_ids[1])
    assert version_info["name"] == "v2" 