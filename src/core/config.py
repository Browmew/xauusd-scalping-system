"""
Configuration Management System

Centralized configuration loading and validation for the XAUUSD scalping system.
Supports YAML configuration files with environment variable substitution,
schema validation, and hot-reloading capabilities.

Time Complexity: O(1) for config access after loading
Space Complexity: O(n) where n is total config size
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, validator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Configuration file paths"""
    data: str = "configs/data_config.yml"
    features: str = "configs/feature_config.yml"
    models: str = "configs/model_config.yml"
    backtest: str = "configs/backtest_config.yml"
    live: str = "configs/live_config.yml"
    risk: str = "configs/risk_config.yml"


class DataConfig(BaseModel):
    """Data configuration schema validation"""
    sources: Dict[str, Any]
    collection: Dict[str, Any]
    quality: Dict[str, Any]
    storage: Dict[str, Any]
    processing: Dict[str, Any]
    validation: Dict[str, Any]
    sessions: Dict[str, Any]
    
    
class FeatureConfig(BaseModel):
    """Feature configuration schema validation"""
    mode: str
    minimal_features: list
    feature_groups: Dict[str, Any]
    parameters: Dict[str, Any]
    realtime: Dict[str, Any]
    selection: Dict[str, Any]
    output: Dict[str, Any]
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['minimal', 'extended']:
            raise ValueError('mode must be either "minimal" or "extended"')
        return v


class ModelConfig(BaseModel):
    """Model configuration schema validation"""
    architecture: Dict[str, Any]
    training: Dict[str, Any]
    target: Dict[str, Any]
    lightgbm: Dict[str, Any]
    xgboost: Dict[str, Any]
    lstm: Dict[str, Any]
    stacking: Dict[str, Any]
    hyperparameter_optimization: Dict[str, Any]
    evaluation: Dict[str, Any]
    deployment: Dict[str, Any]
    explainability: Dict[str, Any]
    persistence: Dict[str, Any]
    optimization: Dict[str, Any]
    logging: Dict[str, Any]


class BacktestConfig(BaseModel):
    """Backtest configuration schema validation"""
    period: Dict[str, Any]
    data: Dict[str, Any]
    execution: Dict[str, Any]
    costs: Dict[str, Any]
    portfolio: Dict[str, Any]
    risk: Dict[str, Any]
    strategy: Dict[str, Any]
    engine: Dict[str, Any]
    metrics: Dict[str, Any]
    benchmarks: list
    stress_tests: Dict[str, Any]
    reporting: Dict[str, Any]
    optimization: Dict[str, Any]
    export: Dict[str, Any]
    debug: Dict[str, Any]


class LiveConfig(BaseModel):
    """Live trading configuration schema validation"""
    mode: str
    broker: Dict[str, Any]
    symbol: Dict[str, Any]
    feeds: Dict[str, Any]
    strategy: Dict[str, Any]
    execution: Dict[str, Any]
    performance: Dict[str, Any]
    risk: Dict[str, Any]
    health: Dict[str, Any]
    logging: Dict[str, Any]
    alerting: Dict[str, Any]
    persistence: Dict[str, Any]
    api: Dict[str, Any]
    development: Dict[str, Any]
    backup: Dict[str, Any]
    compliance: Dict[str, Any]
    disaster_recovery: Dict[str, Any]
    environment: Dict[str, Any]
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['paper', 'live']:
            raise ValueError('mode must be either "paper" or "live"')
        return v


class RiskConfig(BaseModel):
    """Risk configuration schema validation"""
    position_sizing: Dict[str, Any]
    stop_loss: Dict[str, Any]
    take_profit: Dict[str, Any]
    daily_limits: Dict[str, Any]
    periodic_limits: Dict[str, Any]
    correlation: Dict[str, Any]
    concentration: Dict[str, Any]
    drawdown: Dict[str, Any]
    session_risk: Dict[str, Any]
    news_risk: Dict[str, Any]
    volatility_risk: Dict[str, Any]
    liquidity_risk: Dict[str, Any]
    technology_risk: Dict[str, Any]
    model_risk: Dict[str, Any]
    operational_risk: Dict[str, Any]
    regulatory_risk: Dict[str, Any]
    monitoring: Dict[str, Any]
    reporting: Dict[str, Any]
    emergency: Dict[str, Any]
    backtesting: Dict[str, Any]
    calibration: Dict[str, Any]


class ConfigManager:
    """
    Centralized configuration management system
    
    Handles loading, validation, and hot-reloading of all configuration files.
    Provides type-safe access to configuration parameters with environment
    variable substitution and schema validation.
    """
    
    def __init__(self, config_dir: str = "configs", validate: bool = True):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
            validate: Whether to validate configurations against schemas
        """
        self.config_dir = Path(config_dir)
        self.validate = validate
        self._configs: Dict[str, Any] = {}
        self._loaded_at: Dict[str, datetime] = {}
        self._config_schemas = {
            'data': DataConfig,
            'features': FeatureConfig,
            'models': ModelConfig,
            'backtest': BacktestConfig,
            'live': LiveConfig,
            'risk': RiskConfig
        }
        
        # Load all configurations
        self.load_all()
        
    def load_all(self) -> None:
        """Load all configuration files"""
        config_files = {
            'data': 'data_config.yml',
            'features': 'feature_config.yml', 
            'models': 'model_config.yml',
            'backtest': 'backtest_config.yml',
            'live': 'live_config.yml',
            'risk': 'risk_config.yml'
        }
        
        for config_name, filename in config_files.items():
            self.load_config(config_name, filename)
            
    def load_config(self, config_name: str, filename: str) -> None:
        """
        Load a specific configuration file
        
        Args:
            config_name: Name of the configuration
            filename: YAML filename to load
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse YAML
            config_data = yaml.safe_load(content)
            
            # Validate if enabled
            if self.validate and config_name in self._config_schemas:
                schema_class = self._config_schemas[config_name]
                validated_config = schema_class(**config_data)
                config_data = validated_config.dict()
                
            self._configs[config_name] = config_data
            self._loaded_at[config_name] = datetime.now()
            
            logger.info(f"Loaded configuration: {config_name} from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_name}: {e}")
            raise
            
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content
        
        Args:
            content: Raw configuration content
            
        Returns:
            Content with environment variables substituted
        """
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(3) if match.group(3) else ""
            return os.getenv(var_name, default_value)
            
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}]+?)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
        
    def get(self, config_name: str, key_path: str = None, default: Any = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value (e.g., 'data.sources.primary')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if config_name not in self._configs:
            logger.warning(f"Configuration not found: {config_name}")
            return default
            
        config = self._configs[config_name]
        
        if key_path is None:
            return config
            
        # Navigate through nested keys
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            logger.warning(f"Configuration key not found: {config_name}.{key_path}")
            return default
            
    def set(self, config_name: str, key_path: str, value: Any) -> None:
        """
        Set configuration value (runtime only, not persisted)
        
        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value
            value: Value to set
        """
        if config_name not in self._configs:
            self._configs[config_name] = {}
            
        config = self._configs[config_name]
        keys = key_path.split('.')
        
        # Navigate to parent and set value
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
        logger.info(f"Set configuration: {config_name}.{key_path} = {value}")
        
    def reload(self, config_name: str = None) -> None:
        """
        Reload configuration files
        
        Args:
            config_name: Specific config to reload, or None for all
        """
        if config_name:
            # Reload specific config
            config_files = {
                'data': 'data_config.yml',
                'features': 'feature_config.yml',
                'models': 'model_config.yml', 
                'backtest': 'backtest_config.yml',
                'live': 'live_config.yml',
                'risk': 'risk_config.yml'
            }
            
            if config_name in config_files:
                self.load_config(config_name, config_files[config_name])
            else:
                logger.warning(f"Unknown configuration name: {config_name}")
        else:
            # Reload all configs
            self.load_all()
            
    def is_development(self) -> bool:
        """Check if running in development mode"""
        env = self.get('live', 'environment.env', 'production')
        return env in ['development', 'staging']
        
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled"""
        return self.get('live', 'mode', 'paper') == 'paper'
        
    def get_minimal_features(self) -> list:
        """Get list of minimal features for low-latency mode"""
        return self.get('features', 'minimal_features', [])
        
    def get_session_config(self, session_name: str) -> Dict[str, Any]:
        """Get configuration for a specific trading session"""
        return self.get('data', f'sessions.{session_name}', {})
        
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.get('models', model_name, {})
        
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get all risk limits"""
        return {
            'daily': self.get('risk', 'daily_limits', {}),
            'position': self.get('risk', 'position_sizing', {}),
            'drawdown': self.get('risk', 'drawdown', {}),
            'correlation': self.get('risk', 'correlation', {}),
        }
        
    def get_performance_targets(self) -> Dict[str, Any]:
        """Get performance targets for latency and memory"""
        return {
            'latency_ms': self.get('live', 'performance.latency.target_ms', 50),
            'memory_mb': self.get('live', 'performance.memory.target_mb', 200),
            'cpu_pct': self.get('live', 'performance.cpu.target_usage_pct', 50),
        }
        
    def export_config(self, config_name: str, output_path: str) -> None:
        """
        Export configuration to file
        
        Args:
            config_name: Name of configuration to export
            output_path: Path to save the configuration
        """
        if config_name not in self._configs:
            raise ValueError(f"Configuration not found: {config_name}")
            
        with open(output_path, 'w') as f:
            yaml.dump(self._configs[config_name], f, default_flow_style=False)
            
        logger.info(f"Exported configuration {config_name} to {output_path}")
        
    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all loaded configurations
        
        Returns:
            Dictionary mapping config names to validation status
        """
        results = {}
        
        for config_name in self._configs:
            try:
                if config_name in self._config_schemas:
                    schema_class = self._config_schemas[config_name]
                    schema_class(**self._configs[config_name])
                    results[config_name] = True
                    logger.info(f"Configuration validation passed: {config_name}")
                else:
                    results[config_name] = True  # No schema available
                    logger.warning(f"No validation schema for: {config_name}")
            except Exception as e:
                results[config_name] = False
                logger.error(f"Configuration validation failed for {config_name}: {e}")
                
        return results
        
    def get_loaded_configs(self) -> Dict[str, datetime]:
        """Get information about loaded configurations"""
        return self._loaded_at.copy()
        
    def __repr__(self) -> str:
        """String representation"""
        configs = list(self._configs.keys())
        return f"ConfigManager(configs={configs}, validate={self.validate})"


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_dir: str = "configs", validate: bool = True) -> ConfigManager:
    """
    Load configuration and return manager instance
    
    Args:
        config_dir: Directory containing configuration files
        validate: Whether to validate configurations
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir, validate)
    return _config_manager