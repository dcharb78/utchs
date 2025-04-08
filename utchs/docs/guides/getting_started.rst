Getting Started
==============

This guide will help you get started with the UTCHS framework.

Installation
-----------

Install UTCHS using pip:

.. code-block:: bash

   pip install utchs

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/utchs.git
   cd utchs
   pip install -e .

Basic Usage
----------

Here's a simple example of using UTCHS:

.. code-block:: python

   from utchs.core.system import System
   from utchs.math.phase_field import PhaseField
   from utchs.config.config_manager import ConfigManager

   # Load configuration
   config_manager = ConfigManager()
   config = config_manager.get_config()

   # Create a phase field
   field = PhaseField(
       dimensions=config.dimensions,
       lattice_size=config.lattice_size
   )

   # Create and run the system
   system = System(field, config)
   system.run()

Configuration
------------

UTCHS uses a configuration system to manage parameters:

.. code-block:: python

   from utchs.config.config_manager import ConfigManager

   # Create a new configuration
   config_manager = ConfigManager()

   # Update configuration
   config_manager.update_config({
       "dimensions": 2,
       "lattice_size": 50,
       "field_type": "phase"
   })

   # Save configuration
   config_manager.save_config("my_config.yaml")

   # Load configuration
   config_manager = ConfigManager("my_config.yaml")

Model Registry
------------

The model registry helps track and version your models:

.. code-block:: python

   from utchs.utils.model_registry import ModelRegistry

   # Initialize registry
   registry = ModelRegistry()

   # Register a model
   model_id = registry.register_model(
       name="my_model",
       description="A test model",
       parameters={"param1": 1.0}
   )

   # Add a version
   version_id = registry.add_version(
       model_id=model_id,
       version_name="v1",
       data={"field": field},
       metrics={"accuracy": 0.95}
   )

Experiment Tracking
----------------

Track your experiments:

.. code-block:: python

   from utchs.utils.experiment_tracker import ExperimentTracker

   # Initialize tracker
   tracker = ExperimentTracker()

   # Start experiment
   experiment_id = tracker.start_experiment(
       name="test_experiment",
       description="Testing parameters"
   )

   # Log metrics
   tracker.log_metric(experiment_id, "accuracy", 0.95)
   tracker.log_parameter(experiment_id, "learning_rate", 0.01)

   # End experiment
   tracker.end_experiment(experiment_id)

Next Steps
---------

- Check out the :doc:`api/index` for detailed API documentation
- Explore the :doc:`examples/index` for more examples
- Read the :doc:`development/contributing` guide to contribute 