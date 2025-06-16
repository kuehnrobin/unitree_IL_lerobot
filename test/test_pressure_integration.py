#!/usr/bin/env python3
"""
Test script for pressure sensor integration in eval_g1.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pressure_sensor import PressureSensorCollector

def test_pressure_sensor_basic():
    """Test basic functionality of pressure sensor collector"""
    print("Testing PressureSensorCollector...")
    
    # Test with disabled sensor (should work without SDK)
    collector = PressureSensorCollector(enabled=False)
    print("✓ Created disabled pressure sensor collector")
    
    # Test getting data when disabled
    data = collector.get_pressure_data()
    expected_keys = ['left_pressure', 'left_temp', 'right_pressure', 'right_temp']
    
    for key in expected_keys:
        assert key in data, f"Missing key: {key}"
        assert isinstance(data[key], list), f"Data for {key} should be a list"
        assert len(data[key]) == 12, f"Data for {key} should have 12 values"
        assert all(v == 0.0 for v in data[key]), f"Data for {key} should be zeros when disabled"
    
    print("✓ Pressure data format is correct when disabled")
    
    # Test summary stats
    stats = collector.get_summary_stats()
    expected_stats = ['left_max_pressure', 'left_avg_pressure', 'right_max_pressure', 'right_avg_pressure']
    
    for key in expected_stats:
        assert key in stats, f"Missing stat: {key}"
        assert stats[key] == 0.0, f"Stat {key} should be 0.0 when disabled"
    
    print("✓ Summary statistics are correct when disabled")
    
    # Test start/stop (should be no-op when disabled)
    result = collector.start()
    assert result == True, "Start should return True when disabled"
    print("✓ Start method works when disabled")
    
    collector.stop()
    print("✓ Stop method works when disabled")
    
    print("All tests passed! ✓")

def test_pressure_data_format():
    """Test that pressure data format matches what's expected in recording"""
    collector = PressureSensorCollector(enabled=False)
    
    # Simulate the recording integration
    pressure_data = collector.get_pressure_data()
    
    # This is how it's used in the eval_g1.py recording section
    states = {
        "left_hand": {
            "qpos": [1, 2, 3, 4, 5, 6, 7],  # example hand positions
            "qvel": [],
            "torque": [],
            "pressures": pressure_data['left_pressure'],
            "temperatures": pressure_data['left_temp'],
        },
        "right_hand": {
            "qpos": [1, 2, 3, 4, 5, 6, 7],  # example hand positions
            "qvel": [],
            "torque": [],
            "pressures": pressure_data['right_pressure'],
            "temperatures": pressure_data['right_temp'],
        }
    }
    
    # Verify the structure is correct
    assert len(states["left_hand"]["pressures"]) == 12
    assert len(states["left_hand"]["temperatures"]) == 12
    assert len(states["right_hand"]["pressures"]) == 12
    assert len(states["right_hand"]["temperatures"]) == 12
    
    print("✓ Pressure data integration format is correct")

if __name__ == "__main__":
    try:
        test_pressure_sensor_basic()
        test_pressure_data_format()
        print("\n🎉 All pressure sensor tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
