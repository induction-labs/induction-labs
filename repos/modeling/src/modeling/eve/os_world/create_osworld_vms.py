import asyncio
import time
import uuid
from venv import logger

import dotenv
from google.cloud import compute_v1

# load google application credentials from .env file
dotenv.load_dotenv()

# Your existing GCP functions (keeping them for reference)
SERVER_TEMPLATE_NAME = "osworld-annotation-server-us-central1-v4-3"
SERVER_REGION = "us-central1-a"
PROJECT_ID = "induction-labs"


def wait_for_operation(project_id, zone, operation_name, timeout=300):
    """Wait for a zone operation to complete."""
    zone_operations_client = compute_v1.ZoneOperationsClient()
    start_time = time.time()

    while time.time() - start_time < timeout:
        request = compute_v1.GetZoneOperationRequest(
            project=project_id, zone=zone, operation=operation_name
        )

        operation = zone_operations_client.get(request=request)

        if operation.status == compute_v1.Operation.Status.DONE:
            if operation.error:
                raise Exception(f"Operation failed: {operation.error}")
            return operation

        # print(f"Waiting for operation... Status: {operation.status}")
        time.sleep(10)

    raise TimeoutError(
        f"Operation {operation_name} did not complete within {timeout} seconds"
    )


def get_instance_internal_ip(project_id, zone, instance_name):
    """Get the internal IP address of an instance."""
    instances_client = compute_v1.InstancesClient()

    request = compute_v1.GetInstanceRequest(
        project=project_id, zone=zone, instance=instance_name
    )

    instance = instances_client.get(request=request)

    # Get internal IP from the first network interface
    if instance.network_interfaces:
        network_interface = instance.network_interfaces[0]
        return network_interface.network_i_p

    return None


def create_vm_from_template(project_id, zone, instance_name, template_name):
    client = compute_v1.InstancesClient()

    # Create instance with minimal configuration
    instance = compute_v1.Instance()
    instance.name = instance_name

    # The template URL for source_instance_template
    template_url = f"projects/{project_id}/global/instanceTemplates/{template_name}"

    request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance,
        source_instance_template=template_url,
    )

    operation = client.insert(request=request)
    print(f"Creating VM: {operation.name}")
    return operation


def create_new_vm_sync():
    """Synchronous VM creation function"""
    random_id = str(uuid.uuid4())[:8]
    name = f"osworld-annotation-server-v3-{random_id}"

    print(f"Creating VM: {name}")

    # Create the VM
    try:
        operation = create_vm_from_template(
            project_id=PROJECT_ID,
            zone=SERVER_REGION,
            instance_name=name,
            template_name=SERVER_TEMPLATE_NAME,
        )
        print("Waiting for VM creation to complete...")
        wait_for_operation(
            project_id=PROJECT_ID, zone=SERVER_REGION, operation_name=operation.name
        )

        # Get the internal IP
        internal_ip = get_instance_internal_ip(
            project_id=PROJECT_ID, zone=SERVER_REGION, instance_name=name
        )
    except Exception as e:
        print(f"Error creating VM {name}: {e}")
        raise e

    # Wait for the operation to complete

    print(f"VM '{name}' created successfully!")
    print(f"Internal IP: {internal_ip}")

    return {
        "instance_name": name,
        "internal_ip": internal_ip,
        "zone": SERVER_REGION,
        "project_id": PROJECT_ID,
    }


def check_vm_status_sync(project_id, zone, instance_name):
    """Check if a VM is preempted or terminated"""
    instances_client = compute_v1.InstancesClient()

    try:
        request = compute_v1.GetInstanceRequest(
            project=project_id, zone=zone, instance=instance_name
        )

        instance = instances_client.get(request=request)

        # Check if instance is preempted or terminated
        status = instance.status

        # PREEMPTED is when the instance was preempted by GCP
        # TERMINATED is when the instance is stopped/terminated
        is_preempted = status != compute_v1.Instance.Status.RUNNING.name
        print(is_preempted, status, instance_name)

        return {
            "is_preempted": is_preempted,
            "status": status,
            "instance_name": instance_name,
        }

    except Exception as e:
        print(f"Error checking VM status for {instance_name}: {e}")
        # If we can't check the status, assume it's preempted
        return {
            "is_preempted": True,
            "status": "UNKNOWN",
            "instance_name": instance_name,
        }


def delete_vm_sync(project_id, zone, instance_name):
    """Delete a VM instance and wait for the operation to complete."""
    client = compute_v1.InstancesClient()

    print(f"Deleting VM: {instance_name}")

    try:
        # Create the delete request
        request = compute_v1.DeleteInstanceRequest(
            project=project_id, zone=zone, instance=instance_name
        )

        # Execute the delete operation
        operation = client.delete(request=request)
        print(f"Delete operation started: {operation.name}")

        # Wait for the operation to complete
        print("Waiting for VM deletion to complete...")
        wait_for_operation(
            project_id=project_id, zone=zone, operation_name=operation.name
        )

        print(f"VM '{instance_name}' deleted successfully!")

        return {
            "instance_name": instance_name,
            "zone": zone,
            "project_id": project_id,
            "status": "deleted",
        }
    except Exception as e:
        print(f"Error deleting VM {instance_name}: {e}")
        return {
            "instance_name": instance_name,
            "zone": zone,
            "project_id": project_id,
            "status": "error",
            "error": str(e),
        }


class AsyncVMRoundRobin:
    def __init__(self):
        self._vms: list[dict] = []
        self._i = 0
        self._lock = asyncio.Lock()

    async def initialize(self, num_vms: int):
        """Create initial VMs"""
        tasks = [asyncio.to_thread(create_new_vm_sync) for _ in range(num_vms)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if not isinstance(result, Exception):
                self._vms.append(result)

    async def _is_vm_preempted(self, vm_info: dict) -> bool:
        """Check if a VM is preempted"""
        status_info = await asyncio.to_thread(
            check_vm_status_sync, PROJECT_ID, SERVER_REGION, vm_info["instance_name"]
        )
        return status_info["is_preempted"]

    async def _add_replacement_vm(self):
        """Add a replacement VM asynchronously"""
        try:
            new_vm = await asyncio.to_thread(create_new_vm_sync)
            async with self._lock:
                self._vms.append(new_vm)
        except Exception as e:
            logger.error(f"Failed to create replacement VM: {e}")

    async def next(self) -> dict:
        """Get the next available VM, skipping preempted ones"""
        async with self._lock:
            if not self._vms:
                raise Exception("No VMs available")

            for _ in range(len(self._vms)):
                if self._i >= len(self._vms):
                    self._i = 0

                vm = self._vms[self._i]

                if await self._is_vm_preempted(vm):
                    # Remove preempted VM and create replacement
                    removed_vm = self._vms.pop(self._i)
                    if self._i >= len(self._vms) and self._vms:
                        self._i = 0

                    # Create replacement VM
                    asyncio.create_task(self._add_replacement_vm())  # noqa: RUF006

                    # Delete preempted VM (fire and forget)
                    asyncio.create_task(  # noqa: RUF006
                        asyncio.to_thread(
                            delete_vm_sync,
                            PROJECT_ID,
                            SERVER_REGION,
                            removed_vm["instance_name"],
                        )
                    )
                    continue

                # VM is good, return it
                self._i = (self._i + 1) % len(self._vms)
                return vm

            raise Exception("All VMs are preempted")

    async def cleanup(self):
        """Delete all VMs"""
        async with self._lock:
            tasks = [
                asyncio.to_thread(
                    delete_vm_sync, PROJECT_ID, SERVER_REGION, vm["instance_name"]
                )
                for vm in self._vms
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._vms.clear()


# Helper function
async def create_vm_pool(num_vms: int) -> AsyncVMRoundRobin:
    vm_pool = AsyncVMRoundRobin()
    await vm_pool.initialize(num_vms)
    return vm_pool


# Example usage:
async def main():
    vms = await create_vm_pool(num_vms=3)

    try:
        for _ in range(20):
            vm = await vms.next()
            await asyncio.sleep(5)
            print(f"Using VM: {vm['instance_name']} {vm['internal_ip']}")
    finally:
        await vms.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
