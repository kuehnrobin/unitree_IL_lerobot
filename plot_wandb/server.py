"""
WandB server management utilities.

This module provides functionality to start and manage a local wandb server
for viewing logs in the web interface.
"""

import subprocess
import time
import webbrowser
import os
import signal
from pathlib import Path
from typing import Optional, Union
import threading
import socket


class WandbServerManager:
    """
    Manager for local wandb server operations.
    """
    
    def __init__(self, wandb_dir: Union[str, Path], port: int = 8080):
        """
        Initialize the server manager.
        
        Args:
            wandb_dir: Path to the wandb directory
            port: Port to run the server on
        """
        self.wandb_dir = Path(wandb_dir)
        self.port = port
        self.process = None
        self.server_url = f"http://localhost:{port}"
        
    def is_port_available(self, port: int) -> bool:
        """
        Check if a port is available.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def find_available_port(self, start_port: int = 8080, max_attempts: int = 100) -> int:
        """
        Find an available port starting from start_port.
        
        Args:
            start_port: Starting port number
            max_attempts: Maximum number of ports to try
            
        Returns:
            Available port number
            
        Raises:
            RuntimeError: If no available port is found
        """
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                return port
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")
    
    def start_server(self, open_browser: bool = True, sync_tensorboard: bool = False) -> bool:
        """
        Start the local wandb server.
        
        Args:
            open_browser: Whether to automatically open browser
            sync_tensorboard: Whether to sync tensorboard logs
            
        Returns:
            True if server started successfully, False otherwise
        """
        if self.process and self.process.poll() is None:
            print(f"Server already running on {self.server_url}")
            return True
        
        # Find available port if current one is taken
        if not self.is_port_available(self.port):
            self.port = self.find_available_port(self.port)
            self.server_url = f"http://localhost:{self.port}"
            print(f"Port {self.port} was taken, using port {self.port}")
        
        try:
            # Build the command
            cmd = [
                "wandb", "server", "start",
                "--port", str(self.port),
                "--host", "localhost"
            ]
            
            if sync_tensorboard:
                cmd.extend(["--sync-tensorboard"])
            
            # Set environment variable for wandb directory
            env = os.environ.copy()
            env["WANDB_DIR"] = str(self.wandb_dir)
            
            print(f"Starting wandb server on port {self.port}...")
            print(f"Command: {' '.join(cmd)}")
            
            # Start the server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.wandb_dir),
                env=env
            )
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is None:
                print(f"WandB server started successfully!")
                print(f"Access the dashboard at: {self.server_url}")
                
                if open_browser:
                    self._open_browser()
                
                return True
            else:
                stdout, stderr = self.process.communicate()
                print(f"Failed to start server. Error: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            print("wandb command not found. Please install wandb: pip install wandb")
            return False
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        Stop the wandb server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self.process or self.process.poll() is not None:
            print("No server running")
            return True
        
        try:
            # Try graceful shutdown first
            self.process.terminate()
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=10)
                print("Server stopped successfully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                self.process.kill()
                self.process.wait()
                print("Server force stopped")
                return True
                
        except Exception as e:
            print(f"Failed to stop server: {e}")
            return False
    
    def restart_server(self, open_browser: bool = True) -> bool:
        """
        Restart the wandb server.
        
        Args:
            open_browser: Whether to open browser after restart
            
        Returns:
            True if restart successful, False otherwise
        """
        print("Restarting wandb server...")
        self.stop_server()
        time.sleep(2)
        return self.start_server(open_browser)
    
    def _open_browser(self):
        """Open the wandb dashboard in the default browser."""
        def open_browser_delayed():
            time.sleep(2)  # Give server time to fully start
            try:
                webbrowser.open(self.server_url)
                print(f"Opened browser at {self.server_url}")
            except Exception as e:
                print(f"Failed to open browser: {e}")
        
        # Open browser in a separate thread to not block
        thread = threading.Thread(target=open_browser_delayed)
        thread.daemon = True
        thread.start()
    
    def get_status(self) -> dict:
        """
        Get server status information.
        
        Returns:
            Dictionary with status information
        """
        is_running = self.process and self.process.poll() is None
        
        return {
            "running": is_running,
            "port": self.port,
            "url": self.server_url,
            "wandb_dir": str(self.wandb_dir),
            "pid": self.process.pid if is_running else None
        }
    
    def sync_tensorboard_logs(self, tensorboard_log_dir: Union[str, Path]) -> bool:
        """
        Sync tensorboard logs to wandb.
        
        Args:
            tensorboard_log_dir: Path to tensorboard logs
            
        Returns:
            True if sync successful, False otherwise
        """
        try:
            cmd = [
                "wandb", "sync", 
                str(tensorboard_log_dir),
                "--project", "local-training"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.wandb_dir))
            
            if result.returncode == 0:
                print("Successfully synced tensorboard logs")
                return True
            else:
                print(f"Failed to sync logs: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error syncing tensorboard logs: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


def quick_start_server(wandb_dir: Union[str, Path], 
                      port: int = 8080, 
                      open_browser: bool = True) -> WandbServerManager:
    """
    Quick function to start a wandb server.
    
    Args:
        wandb_dir: Path to wandb directory
        port: Port to run server on
        open_browser: Whether to open browser
        
    Returns:
        WandbServerManager instance
    """
    manager = WandbServerManager(wandb_dir, port)
    manager.start_server(open_browser)
    return manager


def find_wandb_runs(base_dir: Union[str, Path]) -> list:
    """
    Find all wandb runs in a directory structure.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of paths to wandb run directories
    """
    base_path = Path(base_dir)
    runs = []
    
    for wandb_file in base_path.rglob("*.wandb"):
        runs.append(wandb_file.parent)
    
    return runs
