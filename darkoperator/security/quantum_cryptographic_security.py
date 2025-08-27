"""
Quantum-Cryptographic Security Framework for Physics Computations.

This module implements advanced security measures incorporating quantum
cryptography, homomorphic encryption, and physics-informed security
protocols specifically designed for sensitive particle physics data.
"""

import torch
import numpy as np
import hashlib
import secrets
import hmac
import base64
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

class SecurityLevel(Enum):
    """Security levels for different types of physics data."""
    PUBLIC = "public"                    # Open research data
    INTERNAL = "internal"               # Lab-internal data
    COLLABORATION = "collaboration"     # Multi-institution research
    SENSITIVE = "sensitive"             # Unpublished results
    CLASSIFIED = "classified"           # Government/defense research

@dataclass
class QuantumSecurityConfig:
    """Configuration for quantum-enhanced security."""
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    quantum_key_length: int = 256
    classical_key_length: int = 4096
    homomorphic_precision: int = 32
    bb84_protocol_enabled: bool = True
    quantum_noise_threshold: float = 0.15
    entanglement_verification: bool = True
    post_quantum_algorithms: List[str] = field(default_factory=lambda: ["CRYSTALS-Kyber", "FALCON"])

class QuantumKeyDistribution:
    """Simulated Quantum Key Distribution using BB84 protocol."""
    
    def __init__(self, key_length: int = 256, noise_threshold: float = 0.15):
        self.key_length = key_length
        self.noise_threshold = noise_threshold
        self.logger = logging.getLogger("QuantumKeyDistribution")
        
    def generate_quantum_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate quantum key pair using simulated BB84 protocol.
        
        Returns:
            Tuple of (alice_key, bob_key) - should be identical if no eavesdropping
        """
        # Simulate quantum bit preparation and measurement
        
        # Alice prepares random bits and bases
        alice_bits = np.random.randint(0, 2, self.key_length * 2)  # Extra bits for sifting
        alice_bases = np.random.randint(0, 2, self.key_length * 2)  # 0: rectilinear, 1: diagonal
        
        # Simulate quantum channel transmission with noise
        received_bits = alice_bits.copy()
        channel_noise = np.random.random(len(alice_bits)) < 0.05  # 5% bit flip probability
        received_bits[channel_noise] = 1 - received_bits[channel_noise]
        
        # Bob randomly chooses measurement bases
        bob_bases = np.random.randint(0, 2, self.key_length * 2)
        
        # Sifting: Keep only bits where Alice and Bob used the same basis
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = received_bits[matching_bases]
        
        # Error estimation: Compare subset of sifted bits
        test_fraction = 0.1
        test_size = int(len(sifted_alice) * test_fraction)
        test_indices = np.random.choice(len(sifted_alice), test_size, replace=False)
        
        test_alice = sifted_alice[test_indices]
        test_bob = sifted_bob[test_indices]
        error_rate = np.mean(test_alice != test_bob)
        
        # Security check
        if error_rate > self.noise_threshold:
            self.logger.warning(f"High error rate detected: {error_rate:.3f} > {self.noise_threshold}")
            raise SecurityError("Potential eavesdropping detected in quantum channel")
        
        # Remove test bits and generate final key
        remaining_indices = np.setdiff1d(np.arange(len(sifted_alice)), test_indices)
        final_alice = sifted_alice[remaining_indices][:self.key_length]
        final_bob = sifted_bob[remaining_indices][:self.key_length]
        
        # Convert to bytes
        alice_key = self._bits_to_bytes(final_alice)
        bob_key = self._bits_to_bytes(final_bob)
        
        self.logger.info(f"Quantum key generated with error rate: {error_rate:.3f}")
        
        return alice_key, bob_key
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        # Pad to multiple of 8
        padded_length = ((len(bits) + 7) // 8) * 8
        padded_bits = np.zeros(padded_length, dtype=int)
        padded_bits[:len(bits)] = bits
        
        # Convert to bytes
        byte_array = []
        for i in range(0, len(padded_bits), 8):
            byte_bits = padded_bits[i:i+8]
            byte_value = sum(bit * (2 ** (7-j)) for j, bit in enumerate(byte_bits))
            byte_array.append(byte_value)
        
        return bytes(byte_array)

class HomomorphicPhysicsEncryption:
    """Homomorphic encryption for physics computations on encrypted data."""
    
    def __init__(self, precision: int = 32):
        self.precision = precision
        self.scaling_factor = 2 ** precision
        self.modulus = 2 ** 64 - 1  # Large prime for arithmetic
        
    def encrypt_tensor(self, tensor: torch.Tensor, key: bytes) -> Dict[str, Any]:
        """
        Encrypt tensor for homomorphic operations.
        
        Args:
            tensor: Input tensor to encrypt
            key: Encryption key
            
        Returns:
            Encrypted tensor representation
        """
        # Convert to fixed-point representation
        scaled_tensor = (tensor * self.scaling_factor).long()
        
        # Generate pseudo-random mask from key
        np.random.seed(int.from_bytes(key[:4], 'big'))
        mask_shape = scaled_tensor.shape
        random_mask = torch.from_numpy(
            np.random.randint(0, self.modulus, size=mask_shape)
        ).long()
        
        # Additive homomorphic encryption
        encrypted_data = (scaled_tensor + random_mask) % self.modulus
        
        # Store metadata
        encryption_metadata = {
            'encrypted_data': encrypted_data,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'scaling_factor': self.scaling_factor,
            'key_hash': hashlib.sha256(key).hexdigest()[:16],
            'timestamp': datetime.now().isoformat()
        }
        
        return encryption_metadata
    
    def decrypt_tensor(self, encrypted_data: Dict[str, Any], key: bytes) -> torch.Tensor:
        """
        Decrypt tensor from homomorphic encryption.
        
        Args:
            encrypted_data: Encrypted tensor representation
            key: Decryption key
            
        Returns:
            Decrypted tensor
        """
        # Verify key
        key_hash = hashlib.sha256(key).hexdigest()[:16]
        if encrypted_data['key_hash'] != key_hash:
            raise SecurityError("Invalid decryption key")
        
        # Regenerate mask
        np.random.seed(int.from_bytes(key[:4], 'big'))
        mask_shape = encrypted_data['shape']
        random_mask = torch.from_numpy(
            np.random.randint(0, self.modulus, size=mask_shape)
        ).long()
        
        # Decrypt
        decrypted_scaled = (encrypted_data['encrypted_data'] - random_mask) % self.modulus
        
        # Convert back to original precision
        decrypted_tensor = decrypted_scaled.float() / self.scaling_factor
        
        return decrypted_tensor
    
    def homomorphic_add(self, encrypted1: Dict[str, Any], 
                       encrypted2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform addition on encrypted tensors."""
        if encrypted1['shape'] != encrypted2['shape']:
            raise ValueError("Tensor shapes must match for homomorphic addition")
        
        result = encrypted1.copy()
        result['encrypted_data'] = (
            encrypted1['encrypted_data'] + encrypted2['encrypted_data']
        ) % self.modulus
        
        return result
    
    def homomorphic_multiply_scalar(self, encrypted: Dict[str, Any], 
                                  scalar: float) -> Dict[str, Any]:
        """Multiply encrypted tensor by scalar."""
        scaled_scalar = int(scalar * self.scaling_factor)
        
        result = encrypted.copy()
        result['encrypted_data'] = (
            encrypted['encrypted_data'] * scaled_scalar
        ) % self.modulus
        
        return result

class PhysicsDataVault:
    """Secure vault for physics data with quantum-enhanced protection."""
    
    def __init__(self, config: QuantumSecurityConfig):
        self.config = config
        self.quantum_kd = QuantumKeyDistribution(
            key_length=config.quantum_key_length,
            noise_threshold=config.quantum_noise_threshold
        )
        self.homomorphic = HomomorphicPhysicsEncryption(config.homomorphic_precision)
        self.vault_data: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("PhysicsDataVault")
        
        # Generate master keys
        self._generate_master_keys()
    
    def _generate_master_keys(self):
        """Generate master encryption keys."""
        try:
            # Quantum key distribution
            self.alice_key, self.bob_key = self.quantum_kd.generate_quantum_key_pair()
            
            # Classical RSA keypair for hybrid encryption
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.classical_key_length,
            )
            self.public_key = self.private_key.public_key()
            
            self.logger.info("Master keys generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate master keys: {e}")
            raise SecurityError(f"Key generation failed: {e}")
    
    def store_physics_data(self, data_id: str, tensor_data: torch.Tensor,
                          metadata: Dict[str, Any],
                          security_level: SecurityLevel = SecurityLevel.INTERNAL) -> str:
        """
        Store physics data with quantum-enhanced security.
        
        Args:
            data_id: Unique identifier for the data
            tensor_data: Physics tensor data to store
            metadata: Associated metadata
            security_level: Required security level
            
        Returns:
            Storage receipt/token
        """
        try:
            # Generate unique session key for this data
            session_key = secrets.token_bytes(32)
            
            # Encrypt tensor using homomorphic encryption
            encrypted_tensor = self.homomorphic.encrypt_tensor(tensor_data, session_key)
            
            # Encrypt session key using quantum-enhanced hybrid encryption
            encrypted_session_key = self._hybrid_encrypt_key(session_key)
            
            # Create secure metadata
            secure_metadata = {
                'original_metadata': metadata,
                'security_level': security_level.value,
                'storage_timestamp': datetime.now().isoformat(),
                'tensor_shape': list(tensor_data.shape),
                'tensor_dtype': str(tensor_data.dtype),
                'checksum': self._compute_tensor_checksum(tensor_data)
            }
            
            # Store in vault
            self.vault_data[data_id] = {
                'encrypted_tensor': encrypted_tensor,
                'encrypted_session_key': encrypted_session_key,
                'secure_metadata': secure_metadata,
                'access_count': 0,
                'last_access': None
            }
            
            # Generate storage receipt
            receipt = self._generate_storage_receipt(data_id, security_level)
            
            # Log access
            self._log_access("STORE", data_id, security_level, success=True)
            
            self.logger.info(f"Physics data stored securely: {data_id}")
            return receipt
            
        except Exception as e:
            self._log_access("STORE", data_id, security_level, success=False, error=str(e))
            raise SecurityError(f"Failed to store data: {e}")
    
    def retrieve_physics_data(self, data_id: str, receipt: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Retrieve and decrypt physics data.
        
        Args:
            data_id: Data identifier
            receipt: Storage receipt for authentication
            
        Returns:
            Tuple of (decrypted_tensor, metadata)
        """
        try:
            # Verify receipt
            if not self._verify_storage_receipt(data_id, receipt):
                raise SecurityError("Invalid storage receipt")
            
            # Check if data exists
            if data_id not in self.vault_data:
                raise SecurityError("Data not found in vault")
            
            vault_entry = self.vault_data[data_id]
            
            # Decrypt session key
            session_key = self._hybrid_decrypt_key(vault_entry['encrypted_session_key'])
            
            # Decrypt tensor
            decrypted_tensor = self.homomorphic.decrypt_tensor(
                vault_entry['encrypted_tensor'], session_key
            )
            
            # Verify integrity
            computed_checksum = self._compute_tensor_checksum(decrypted_tensor)
            stored_checksum = vault_entry['secure_metadata']['checksum']
            
            if computed_checksum != stored_checksum:
                raise SecurityError("Data integrity check failed")
            
            # Update access tracking
            vault_entry['access_count'] += 1
            vault_entry['last_access'] = datetime.now().isoformat()
            
            # Log access
            security_level = SecurityLevel(vault_entry['secure_metadata']['security_level'])
            self._log_access("RETRIEVE", data_id, security_level, success=True)
            
            self.logger.info(f"Physics data retrieved successfully: {data_id}")
            
            return decrypted_tensor, vault_entry['secure_metadata']['original_metadata']
            
        except Exception as e:
            self._log_access("RETRIEVE", data_id, SecurityLevel.INTERNAL, success=False, error=str(e))
            raise SecurityError(f"Failed to retrieve data: {e}")
    
    def _hybrid_encrypt_key(self, session_key: bytes) -> bytes:
        """Encrypt session key using hybrid quantum-classical encryption."""
        # XOR with quantum key
        quantum_encrypted = bytes(
            a ^ b for a, b in zip(session_key, self.alice_key[:len(session_key)])
        )
        
        # Further encrypt with classical RSA
        classical_encrypted = self.public_key.encrypt(
            quantum_encrypted,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return classical_encrypted
    
    def _hybrid_decrypt_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt session key using hybrid quantum-classical decryption."""
        # Decrypt with classical RSA
        quantum_encrypted = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # XOR with quantum key to get original
        session_key = bytes(
            a ^ b for a, b in zip(quantum_encrypted, self.bob_key[:len(quantum_encrypted)])
        )
        
        return session_key
    
    def _compute_tensor_checksum(self, tensor: torch.Tensor) -> str:
        """Compute integrity checksum for tensor."""
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    def _generate_storage_receipt(self, data_id: str, security_level: SecurityLevel) -> str:
        """Generate tamper-proof storage receipt."""
        receipt_data = {
            'data_id': data_id,
            'security_level': security_level.value,
            'timestamp': datetime.now().isoformat(),
            'vault_signature': 'physics_vault_v1'
        }
        
        receipt_json = json.dumps(receipt_data, sort_keys=True)
        receipt_hash = hmac.new(
            self.alice_key[:32], 
            receipt_json.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        receipt = base64.b64encode(
            json.dumps({
                'data': receipt_data,
                'signature': receipt_hash
            }).encode()
        ).decode()
        
        return receipt
    
    def _verify_storage_receipt(self, data_id: str, receipt: str) -> bool:
        """Verify storage receipt authenticity."""
        try:
            receipt_json = base64.b64decode(receipt).decode()
            receipt_obj = json.loads(receipt_json)
            
            # Verify signature
            expected_hash = hmac.new(
                self.alice_key[:32],
                json.dumps(receipt_obj['data'], sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            
            if receipt_obj['signature'] != expected_hash:
                return False
            
            # Verify data_id matches
            if receipt_obj['data']['data_id'] != data_id:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _log_access(self, operation: str, data_id: str, 
                   security_level: SecurityLevel, success: bool,
                   error: Optional[str] = None):
        """Log access attempt."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'data_id': data_id,
            'security_level': security_level.value,
            'success': success,
            'error': error
        }
        
        self.access_log.append(log_entry)
        
        # Log to system logger
        level = logging.INFO if success else logging.WARNING
        msg = f"{operation} {data_id} ({security_level.value}): {'SUCCESS' if success else 'FAILED'}"
        if error:
            msg += f" - {error}"
        self.logger.log(level, msg)
    
    def get_security_audit(self, days_back: int = 7) -> Dict[str, Any]:
        """Get security audit report."""
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        recent_logs = [
            log for log in self.access_log 
            if datetime.fromisoformat(log['timestamp']) >= cutoff_time
        ]
        
        audit_report = {
            'audit_period_days': days_back,
            'total_operations': len(recent_logs),
            'successful_operations': sum(1 for log in recent_logs if log['success']),
            'failed_operations': sum(1 for log in recent_logs if not log['success']),
            'operations_by_type': {},
            'security_levels_accessed': {},
            'stored_data_count': len(self.vault_data),
            'quantum_key_status': 'active',
            'last_audit_time': datetime.now().isoformat()
        }
        
        # Count by operation type
        for log in recent_logs:
            op_type = log['operation']
            audit_report['operations_by_type'][op_type] = audit_report['operations_by_type'].get(op_type, 0) + 1
        
        # Count by security level
        for log in recent_logs:
            sec_level = log['security_level']
            audit_report['security_levels_accessed'][sec_level] = audit_report['security_levels_accessed'].get(sec_level, 0) + 1
        
        return audit_report

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

def create_quantum_security_demo() -> Dict[str, Any]:
    """Create a demonstration of quantum-enhanced security."""
    
    # Configuration
    config = QuantumSecurityConfig(
        security_level=SecurityLevel.SENSITIVE,
        quantum_key_length=128,  # Smaller for demo
        bb84_protocol_enabled=True,
        entanglement_verification=True
    )
    
    # Initialize secure vault
    vault = PhysicsDataVault(config)
    
    # Create test physics data
    physics_tensor = torch.randn(100, 50) * 1000  # Simulated physics measurements
    metadata = {
        'experiment': 'dark_matter_search',
        'detector': 'CMS',
        'run_number': 12345,
        'energy_tev': 13.0,
        'luminosity': 150.0
    }
    
    try:
        # Store data securely
        print("Storing physics data with quantum security...")
        receipt = vault.store_physics_data(
            data_id="cms_dark_matter_run_12345",
            tensor_data=physics_tensor,
            metadata=metadata,
            security_level=SecurityLevel.SENSITIVE
        )
        
        # Retrieve data
        print("Retrieving physics data...")
        retrieved_tensor, retrieved_metadata = vault.retrieve_physics_data(
            data_id="cms_dark_matter_run_12345",
            receipt=receipt
        )
        
        # Verify integrity
        data_integrity_ok = torch.allclose(physics_tensor, retrieved_tensor, atol=1e-6)
        metadata_integrity_ok = retrieved_metadata == metadata
        
        # Test homomorphic operations
        print("Testing homomorphic encryption...")
        test_tensor = torch.randn(10, 10)
        encrypted = vault.homomorphic.encrypt_tensor(test_tensor, vault.alice_key)
        decrypted = vault.homomorphic.decrypt_tensor(encrypted, vault.alice_key)
        homomorphic_ok = torch.allclose(test_tensor, decrypted, atol=1e-3)
        
        # Get security audit
        audit_report = vault.get_security_audit(days_back=1)
        
        demo_results = {
            'quantum_security_initialized': True,
            'data_storage_successful': True,
            'data_retrieval_successful': True,
            'data_integrity_verified': data_integrity_ok,
            'metadata_integrity_verified': metadata_integrity_ok,
            'homomorphic_encryption_verified': homomorphic_ok,
            'audit_report': audit_report,
            'security_level': config.security_level.value,
            'demo_successful': True
        }
        
    except Exception as e:
        demo_results = {
            'demo_successful': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
    
    return demo_results

if __name__ == "__main__":
    # Run demonstration
    demo_results = create_quantum_security_demo()
    print("\n✅ Quantum Cryptographic Security Demo Results:")
    print(f"Demo Successful: {demo_results.get('demo_successful', False)}")
    
    if demo_results.get('demo_successful', False):
        print(f"Data Integrity Verified: {demo_results['data_integrity_verified']}")
        print(f"Homomorphic Encryption Verified: {demo_results['homomorphic_encryption_verified']}")
        print(f"Security Operations: {demo_results['audit_report']['total_operations']}")
        print(f"Security Level: {demo_results['security_level']}")
    else:
        print(f"Demo Failed: {demo_results.get('error', 'Unknown error')}")