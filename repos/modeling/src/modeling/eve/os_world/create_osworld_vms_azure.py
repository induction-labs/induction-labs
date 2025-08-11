import asyncio
import json
import logging
import os
import uuid

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    HardwareProfile,
    ImageReference,
    NetworkInterfaceReference,
    NetworkProfile,
    OSDisk,
    SecurityProfile,
    StorageProfile,
    UefiSettings,
    VirtualMachine,
)
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.network.models import NetworkInterface, NetworkInterfaceIPConfiguration

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
# Or disable Azure HTTP logging if you turned it on:
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

with open(os.environ.get("AZURE_SECRET_KEY", "/secrets/azure_creds.json")) as f:
    SECRETS = json.load(f)
os.environ["AZURE_CLIENT_ID"] = SECRETS["clientId"]
os.environ["AZURE_CLIENT_SECRET"] = SECRETS["clientSecret"]
os.environ["AZURE_SUBSCRIPTION_ID"] = SECRETS["subscriptionId"]
os.environ["AZURE_TENANT_ID"] = SECRETS["tenantId"]

# ===========
# CONFIG
# ===========
SUBSCRIPTION_ID = os.environ.get(
    "AZURE_SUBSCRIPTION_ID", "edceb3af-218f-4c5e-b76d-0af8eeda6bb6"
)
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP", "osworld-evals")
LOCATION = os.environ.get("AZ_LOCATION", "northcentralus")

# Networking (must already exist)
VNET_NAME = os.environ.get("AZ_VNET_NAME", "osworld-evals")
SUBNET_NAME = os.environ.get("AZ_SUBNET_NAME", "osworld-eval-notes")

# VM base settings
VM_SIZE = os.environ.get("AZ_VM_SIZE", "Standard_D8as_v5")
ADMIN_USERNAME = os.environ.get("AZ_ADMIN_USERNAME", "azureuser")
PUB_KEY = """
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDQeNOtQisQ/lVCtUFzg4Bz4mZa36Mvr6Y6HVUzPGl+QHDSpQUKBXo39FwpnDdH4ewaemUSz+m/o+1kT2HHktwQ6+0a9TG6OWum2VZkkNfGqNuzXD6kYzuF9IyvEExQXHHsxsq/DejztanO0FWgTlT8yn74epkN55xgHupsFRSJXNX5pyEkOQMKQIaOOMiWiDv6BWB2jivWM509yBSpRD5tAcz6pZYR6Xe1Z42gDvKHd30i0uV2gVlSUcsojeC/nu6Vn+0J+QiqkWbUv+beErZxGyH1DZfGnlfclOt/hDVUcOQQVLqhphJYer8HX2pNbBYEu9iTuiTmIlSSxpawzAMZcLvU3S++JysGHHshV8yzFU27YQDo1m/1FfWMoIbOyE+vXsk5O4nn5ahTnbQiNzFQRIkQSdK7PygcXndB0ke/W3nZn6vN1v/Nze28BIo8Jo9yXjtcxsde8gu435zUFo5x46swQIzV8/Z4Rd+gZgGrhers85/2l82JSvda2GZ53t0= li.jonathan42@gmail.com
""".strip()
SSH_PUB_KEY = os.environ.get("AZ_SSH_PUB_KEY", PUB_KEY)

# IMAGE SOURCE
# Option A (recommended): Azure Compute Gallery image **version** resource ID, e.g.:
#   /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Compute/galleries/<gallery>/images/<imageDef>/versions/1.0.0
GALLERY_IMAGE_VERSION_ID = os.environ.get(
    "AZ_GALLERY_IMAGE_VERSION_ID",
    "/subscriptions/edceb3af-218f-4c5e-b76d-0af8eeda6bb6/resourceGroups/osworld-evals/providers/Microsoft.Compute/galleries/osworld_north_us_central/images/osworld-image/versions/0.0.1",
)

# Option B: Managed Image resource ID (set if not using gallery)
# MANAGED_IMAGE_ID = os.environ.get("AZ_MANAGED_IMAGE_ID", "")

# ===========
# CLIENTS
# ===========
credential = DefaultAzureCredential()
compute = ComputeManagementClient(credential, SUBSCRIPTION_ID)
network = NetworkManagementClient(credential, SUBSCRIPTION_ID)


def _subnet_id() -> str:
    subnet = network.subnets.get(RESOURCE_GROUP, VNET_NAME, SUBNET_NAME)
    return subnet.id


def _create_nic(nic_name: str) -> str:
    ip_cfg = NetworkInterfaceIPConfiguration(
        name=f"{nic_name}-ipcfg",
        subnet={"id": _subnet_id()},
        private_ip_allocation_method="Dynamic",
        # no public IP
    )
    poller = network.network_interfaces.begin_create_or_update(
        RESOURCE_GROUP,
        nic_name,
        NetworkInterface(
            location=LOCATION,
            ip_configurations=[ip_cfg],
            enable_accelerated_networking=True,  # optional, depends on VM size
            # delete_option="Delete",
        ),
    )
    nic = poller.result()
    return nic.id


def _image_reference() -> ImageReference:
    if GALLERY_IMAGE_VERSION_ID:
        # Use a Shared (Azure Compute) Gallery image version
        return ImageReference(id=GALLERY_IMAGE_VERSION_ID)
    # if MANAGED_IMAGE_ID:
    #     # Use a Managed Image
    #     return ImageReference(id=MANAGED_IMAGE_ID)
    raise ValueError("Set AZ_GALLERY_IMAGE_VERSION_ID or AZ_MANAGED_IMAGE_ID")


def _create_vm_from_image(vm_name: str) -> dict:
    nic_name = f"{vm_name}-nic"
    nic_id = _create_nic(nic_name)

    vm = VirtualMachine(
        location=LOCATION,
        hardware_profile=HardwareProfile(vm_size=VM_SIZE),
        storage_profile=StorageProfile(
            image_reference=_image_reference(),
            os_disk=OSDisk(
                name=f"{vm_name}-osdisk",
                caching="ReadWrite",
                create_option="FromImage",
                managed_disk={"storageAccountType": "Premium_LRS"},
                delete_option="Delete",
            ),
        ),
        # os_profile=OSProfile(
        #     computer_name=vm_name,
        #     admin_username=ADMIN_USERNAME,
        #     linux_configuration=LinuxConfiguration(
        #         disable_password_authentication=True,
        #         ssh=SshConfiguration(
        #             public_keys=[
        #                 SshPublicKey(path=f"/home/{ADMIN_USERNAME}/.ssh/authorized_keys", key_data=SSH_PUB_KEY)
        #             ]
        #         ),
        #     ),
        # ),
        security_profile=SecurityProfile(
            security_type="TrustedLaunch",
            uefi_settings=UefiSettings(secure_boot_enabled=True, v_tpm_enabled=True),
        ),
        network_profile=NetworkProfile(
            network_interfaces=[
                NetworkInterfaceReference(
                    id=nic_id, primary=True, delete_option="Delete"
                )
            ]
        ),
    )

    op = compute.virtual_machines.begin_create_or_update(RESOURCE_GROUP, vm_name, vm)
    op.result()  # wait

    # Get private IP
    nic = network.network_interfaces.get(RESOURCE_GROUP, nic_name)
    private_ip = nic.ip_configurations[0].private_ip_address

    return {
        "instance_name": vm_name,
        "internal_ip": private_ip,
        "resource_group": RESOURCE_GROUP,
        "nic_name": nic_name,
    }


def create_new_vm_sync() -> dict:
    vm_name = f"osworld-annotation-az-{str(uuid.uuid4())[:8]}"
    try:
        return _create_vm_from_image(vm_name)
    except Exception as e:
        raise RuntimeError(f"Error creating VM {vm_name}: {e}") from e


def delete_vm_sync(instance_name: str, nic_name: str) -> dict:
    # Delete VM (keeps disks unless we set delete options; this keeps things explicit)
    vm_del = compute.virtual_machines.begin_delete(RESOURCE_GROUP, instance_name)
    vm_del.result()

    # Optionally also delete the OS disk to avoid orphaned disks & costs
    # Get the VM's OS disk name by reading the managed disk (if needed)
    # (If you prefer auto-delete with the VM, set delete options on the NIC/OS disk at create time.)

    # Delete NIC
    nic_del = network.network_interfaces.begin_delete(RESOURCE_GROUP, nic_name)
    nic_del.result()

    return {"instance_name": instance_name, "status": "deleted"}


async def create_new_vm():
    res = await asyncio.to_thread(create_new_vm_sync)
    # wait for vm to boot
    await asyncio.sleep(120)
    return res


class AsyncVMRoundRobin:
    def __init__(self):
        self._vms: list[dict] = []
        self._i = 0
        self._lock = asyncio.Lock()

    async def initialize(self, num_vms: int):
        tasks = [create_new_vm() for _ in range(num_vms)]
        results = await asyncio.gather(*tasks)
        for r in results:
            if not isinstance(r, Exception):
                self._vms.append(r)

    async def next(self) -> dict:
        async with self._lock:
            if not self._vms:
                raise RuntimeError("No VMs available")
            vm = self._vms[self._i]
            self._i = (self._i + 1) % len(self._vms)
            return vm

    async def cleanup(self):
        async with self._lock:
            tasks = [
                asyncio.to_thread(delete_vm_sync, vm["instance_name"], vm["nic_name"])
                for vm in self._vms
            ]
            if tasks:
                await asyncio.gather(*tasks)
            self._vms.clear()


async def create_vm_pool(num_vms: int) -> AsyncVMRoundRobin:
    pool = AsyncVMRoundRobin()
    await pool.initialize(num_vms)
    return pool


# Example usage:
async def main():
    vms = await create_vm_pool(num_vms=1)
    try:
        for _ in range(10):
            vm = await vms.next()
            print(f"Using VM: {vm['instance_name']} {vm['internal_ip']}")
            await asyncio.sleep(2)
    finally:
        await vms.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
