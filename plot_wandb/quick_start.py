#!/usr/bin/env python3
"""
Quick start script for wandb plotting.

This script provides a simple interface to get started with plotting wandb logs.
"""

import sys
from pathlib import Path
import argparse


def quick_plot(wandb_dir: str, output_dir: str = "plots"):
    """Quick plotting function."""
    print("=== WandB Quick Plot ===")
    print(f"Input: {wandb_dir}")
    print(f"Output: {output_dir}")
    
    # Import the main function
    sys.path.insert(0, str(Path(__file__).parent))
    from main import create_training_plots
    
    try:
        create_training_plots(wandb_dir, output_dir, "thesis")
        print(f"\n✅ Plots created successfully in {output_dir}/")
        print("📁 Check the README.md file for usage instructions")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure the wandb directory contains valid runs")
        print("2. Try running: python advanced_extractor.py <wandb_run_dir>")
        print("3. Check if all dependencies are installed: pip install -r requirements.txt")


def quick_server(wandb_dir: str, port: int = 8080):
    """Quick server function."""
    print("=== WandB Quick Server ===")
    print(f"Starting server for: {wandb_dir}")
    
    sys.path.insert(0, str(Path(__file__).parent))
    from server import quick_start_server
    
    try:
        server = quick_start_server(wandb_dir, port)
        print("✅ Server started successfully!")
        print("Press Ctrl+C to stop the server")
        
        import time
        while True:
            time.sleep(1)
            if server.process and server.process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if 'server' in locals():
            server.stop_server()
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick start for wandb plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick plot generation
  python quick_start.py plot /path/to/wandb

  # Start server
  python quick_start.py server /path/to/wandb

  # Plot with custom output
  python quick_start.py plot /path/to/wandb --output my_plots/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Create plots quickly")
    plot_parser.add_argument("wandb_dir", help="Path to wandb directory")
    plot_parser.add_argument("--output", "-o", default="plots", help="Output directory")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start server quickly")
    server_parser.add_argument("wandb_dir", help="Path to wandb directory")
    server_parser.add_argument("--port", "-p", type=int, default=8080, help="Port")
    
    args = parser.parse_args()
    
    if not args.command:
        # Interactive mode
        print("🚀 WandB Plotting Quick Start")
        print()
        print("What would you like to do?")
        print("1. Create plots from wandb logs")
        print("2. Start wandb server")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                wandb_dir = input("Enter wandb directory path: ").strip()
                output_dir = input("Enter output directory (default: plots): ").strip() or "plots"
                quick_plot(wandb_dir, output_dir)
                break
            elif choice == "2":
                wandb_dir = input("Enter wandb directory path: ").strip()
                port = input("Enter port (default: 8080): ").strip()
                port = int(port) if port else 8080
                quick_server(wandb_dir, port)
                break
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
    else:
        # Command line mode
        if args.command == "plot":
            quick_plot(args.wandb_dir, args.output)
        elif args.command == "server":
            quick_server(args.wandb_dir, args.port)


if __name__ == "__main__":
    main()
