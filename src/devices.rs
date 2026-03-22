use anyhow::{bail, Result};
use cpal::traits::{DeviceTrait, HostTrait};

pub fn get_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices()?;
    let mut device_names = Vec::new();
    for device in devices {
        device_names.push(device.name()?);
    }
    Ok(device_names)
}

pub fn get_output_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.output_devices()?;
    let mut device_names = Vec::new();
    for device in devices {
        device_names.push(device.name()?);
    }
    Ok(device_names)
}

pub fn find_cable() -> Result<String> {
    let host = cpal::default_host();
    let cable_needle = "cable output";

    for avail_dev in host.input_devices()? {
        let name = avail_dev.name()?;
        if name.to_lowercase().contains(cable_needle) {
            return Ok(name);
        }
    }

    bail!("Could not find an input device containing 'cable output'")
}

// TODO: set the output device lastly selected by the user in appdata to minimize friction
pub fn set_output_device() -> Result<String> {
    let host = cpal::default_host();

    // Priority order: first match wins.
    // Matching is case-insensitive substring match against the device name.
    let preferred_devices = ["headphones", "speakers"];

    let available = get_output_devices()?;
    for preferred in preferred_devices {
        let needle = preferred.to_lowercase();
        if let Some(name) = available
            .iter()
            .find(|name| name.to_lowercase().contains(&needle))
        {
            return Ok(name.clone());
        }
    }

    bail!("Could not pick an output device.")
}

pub fn device_exists(device: Option<&str>, device_list: &[String]) -> bool {
    match device {
        Some(needle) => device_list
            .iter()
            .any(|d| d.to_lowercase() == needle.to_lowercase()),
        None => false,
    }
}
