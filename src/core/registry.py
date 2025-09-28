"""
Registry pattern implementation for component management.

This module provides a centralized registry system for managing models, 
processors, and other components in a scalable and extensible way.
"""

from typing import Any, Dict, Type, TypeVar, Callable, Optional, List
from abc import ABC, abstractmethod
import inspect
import functools
from pathlib import Path

from .exceptions import RegistryError
from .logger import Logger

T = TypeVar('T')


class ComponentRegistry:
    """
    A centralized registry for managing ML components.
    
    This registry follows the Registry pattern and provides:
    - Component registration and discovery
    - Lazy loading and caching
    - Dependency injection
    - Component lifecycle management
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        self._components: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._logger = logger
        self._initialized = False
        
    def register(
        self, 
        name: str, 
        component_class: Type[T], 
        category: str = "default",
        dependencies: Optional[List[str]] = None,
        factory: Optional[Callable] = None,
        singleton: bool = True
    ) -> None:
        """
        Register a component in the registry.
        
        Args:
            name: Unique name for the component
            component_class: The component class to register
            category: Category for grouping components
            dependencies: List of dependency names
            factory: Custom factory function for creating instances
            singleton: Whether to create singleton instances
        """
        if category not in self._components:
            self._components[category] = {}
            
        if name in self._components[category]:
            if self._logger:
                self._logger.warning(f"Component '{name}' already registered in category '{category}', overwriting")
        
        self._components[category][name] = {
            'class': component_class,
            'dependencies': dependencies or [],
            'factory': factory,
            'singleton': singleton,
            'metadata': {
                'module': component_class.__module__,
                'file': inspect.getfile(component_class),
                'line': inspect.getsourcelines(component_class)[1]
            }
        }
        
        if self._logger:
            self._logger.debug(f"Registered component '{name}' in category '{category}'")
    
    def get(self, name: str, category: str = "default", **kwargs) -> Any:
        """
        Get a component instance from the registry.
        
        Args:
            name: Component name
            category: Component category
            **kwargs: Additional arguments for component creation
            
        Returns:
            Component instance
            
        Raises:
            RegistryError: If component not found or creation fails
        """
        if category not in self._components or name not in self._components[category]:
            raise RegistryError(f"Component '{name}' not found in category '{category}'")
        
        component_info = self._components[category][name]
        
        # Check if singleton instance exists
        instance_key = f"{category}.{name}"
        if component_info['singleton'] and instance_key in self._instances:
            return self._instances[instance_key]
        
        # Create new instance
        try:
            instance = self._create_instance(component_info, **kwargs)
            
            if component_info['singleton']:
                self._instances[instance_key] = instance
                
            if self._logger:
                self._logger.debug(f"Created instance of component '{name}' in category '{category}'")
                
            return instance
            
        except Exception as e:
            raise RegistryError(f"Failed to create component '{name}': {e}") from e
    
    def _create_instance(self, component_info: Dict[str, Any], **kwargs) -> Any:
        """Create a component instance with dependency injection."""
        component_class = component_info['class']
        dependencies = component_info['dependencies']
        factory = component_info['factory']
        
        # Resolve dependencies
        resolved_deps = {}
        for dep_name in dependencies:
            # Try to find dependency in any category
            dep_instance = None
            for cat in self._components:
                if dep_name in self._components[cat]:
                    dep_instance = self.get(dep_name, cat)
                    break
            
            if dep_instance is None:
                raise RegistryError(f"Dependency '{dep_name}' not found for component")
            
            resolved_deps[dep_name] = dep_instance
        
        # Merge dependencies with kwargs
        all_kwargs = {**resolved_deps, **kwargs}
        
        # Create instance using factory or constructor
        if factory:
            return factory(**all_kwargs)
        else:
            # Filter kwargs to only include valid constructor parameters
            sig = inspect.signature(component_class.__init__)
            valid_kwargs = {k: v for k, v in all_kwargs.items() 
                          if k in sig.parameters}
            return component_class(**valid_kwargs)
    
    def list_components(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List all registered components."""
        if category:
            return self._components.get(category, {})
        return self._components
    
    def unregister(self, name: str, category: str = "default") -> None:
        """Unregister a component."""
        if category in self._components and name in self._components[category]:
            del self._components[category][name]
            # Remove singleton instance if exists
            instance_key = f"{category}.{name}"
            if instance_key in self._instances:
                del self._instances[instance_key]
            if self._logger:
                self._logger.debug(f"Unregistered component '{name}' from category '{category}'")
    
    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()
        self._instances.clear()
        self._factories.clear()
        self._dependencies.clear()
        if self._logger:
            self._logger.debug("Registry cleared")


# Global registry instance
_registry = ComponentRegistry()


def register_component(
    name: str, 
    category: str = "default",
    dependencies: Optional[List[str]] = None,
    factory: Optional[Callable] = None,
    singleton: bool = True
):
    """
    Decorator for registering components.
    
    Usage:
        @register_component("my_model", "models")
        class MyModel:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        _registry.register(name, cls, category, dependencies, factory, singleton)
        return cls
    return decorator


def get_component(name: str, category: str = "default", **kwargs) -> Any:
    """Get a component from the global registry."""
    return _registry.get(name, category, **kwargs)


def list_components(category: Optional[str] = None) -> Dict[str, Any]:
    """List components in the global registry."""
    return _registry.list_components(category)


def set_registry_logger(logger: Logger) -> None:
    """Set logger for the global registry."""
    global _registry
    _registry._logger = logger
