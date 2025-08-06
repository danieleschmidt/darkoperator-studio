"""
Internationalization and localization manager for quantum task planning.

Provides multi-language support, regional formatting, and physics-specific
localization for scientific units and notation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import gettext
import locale
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class LocaleInfo:
    """Information about a specific locale."""
    
    code: str  # ISO 639-1 language code
    name: str  # Native language name
    english_name: str  # English language name
    region: str = ""  # ISO 3166-1 alpha-2 country code
    rtl: bool = False  # Right-to-left text direction
    
    # Formatting preferences
    decimal_separator: str = "."
    thousands_separator: str = ","
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    currency_symbol: str = "$"
    
    # Scientific notation preferences
    scientific_notation: bool = True
    use_unicode_symbols: bool = True
    physics_unit_system: str = "SI"  # SI, CGS, Imperial


class PhysicsTranslationManager:
    """Manages physics-specific translations and units."""
    
    def __init__(self):
        self.unit_translations: Dict[str, Dict[str, str]] = {}
        self.physics_terms: Dict[str, Dict[str, str]] = {}
        self.load_physics_translations()
    
    def load_physics_translations(self):
        """Load physics-specific translations."""
        
        # Base SI units translations
        self.unit_translations = {
            "en": {
                "m": "meter", "kg": "kilogram", "s": "second", "A": "ampere",
                "K": "kelvin", "mol": "mole", "cd": "candela",
                "GeV": "gigaelectronvolt", "TeV": "teraelectronvolt", "MeV": "megaelectronvolt",
                "c": "speed of light", "ℏ": "reduced Planck constant", "e": "elementary charge"
            },
            "es": {
                "m": "metro", "kg": "kilogramo", "s": "segundo", "A": "amperio",
                "K": "kelvin", "mol": "mol", "cd": "candela",
                "GeV": "gigaelectronvoltio", "TeV": "teraelectronvoltio", "MeV": "megaelectronvoltio",
                "c": "velocidad de la luz", "ℏ": "constante reducida de Planck", "e": "carga elemental"
            },
            "fr": {
                "m": "mètre", "kg": "kilogramme", "s": "seconde", "A": "ampère",
                "K": "kelvin", "mol": "mole", "cd": "candela",
                "GeV": "gigaélectronvolt", "TeV": "téraélectronvolt", "MeV": "mégaélectronvolt",
                "c": "vitesse de la lumière", "ℏ": "constante réduite de Planck", "e": "charge élémentaire"
            },
            "de": {
                "m": "Meter", "kg": "Kilogramm", "s": "Sekunde", "A": "Ampere",
                "K": "Kelvin", "mol": "Mol", "cd": "Candela",
                "GeV": "Gigaelektronenvolt", "TeV": "Teraelektronenvolt", "MeV": "Megaelektronenvolt",
                "c": "Lichtgeschwindigkeit", "ℏ": "reduzierte Plancksche Konstante", "e": "Elementarladung"
            },
            "ja": {
                "m": "メートル", "kg": "キログラム", "s": "秒", "A": "アンペア",
                "K": "ケルビン", "mol": "モル", "cd": "カンデラ",
                "GeV": "ギガ電子ボルト", "TeV": "テラ電子ボルト", "MeV": "メガ電子ボルト",
                "c": "光速", "ℏ": "換算プランク定数", "e": "素電荷"
            },
            "zh": {
                "m": "米", "kg": "千克", "s": "秒", "A": "安培",
                "K": "开尔文", "mol": "摩尔", "cd": "坎德拉",
                "GeV": "吉电子伏特", "TeV": "太电子伏特", "MeV": "兆电子伏特",
                "c": "光速", "ℏ": "约化普朗克常数", "e": "基本电荷"
            }
        }
        
        # Physics terms translations
        self.physics_terms = {
            "en": {
                "energy": "energy", "momentum": "momentum", "mass": "mass", "velocity": "velocity",
                "particle": "particle", "photon": "photon", "electron": "electron", "proton": "proton",
                "neutron": "neutron", "quark": "quark", "lepton": "lepton", "boson": "boson",
                "conservation": "conservation", "symmetry": "symmetry", "invariance": "invariance",
                "quantum": "quantum", "classical": "classical", "relativistic": "relativistic",
                "detector": "detector", "calorimeter": "calorimeter", "tracker": "tracker",
                "collision": "collision", "interaction": "interaction", "decay": "decay",
                "cross_section": "cross section", "luminosity": "luminosity", "efficiency": "efficiency"
            },
            "es": {
                "energy": "energía", "momentum": "momento", "mass": "masa", "velocity": "velocidad",
                "particle": "partícula", "photon": "fotón", "electron": "electrón", "proton": "protón",
                "neutron": "neutrón", "quark": "quark", "lepton": "leptón", "boson": "bosón",
                "conservation": "conservación", "symmetry": "simetría", "invariance": "invariancia",
                "quantum": "cuántico", "classical": "clásico", "relativistic": "relativista",
                "detector": "detector", "calorimeter": "calorímetro", "tracker": "trazador",
                "collision": "colisión", "interaction": "interacción", "decay": "desintegración",
                "cross_section": "sección eficaz", "luminosity": "luminosidad", "efficiency": "eficiencia"
            },
            "fr": {
                "energy": "énergie", "momentum": "quantité de mouvement", "mass": "masse", "velocity": "vitesse",
                "particle": "particule", "photon": "photon", "electron": "électron", "proton": "proton",
                "neutron": "neutron", "quark": "quark", "lepton": "lepton", "boson": "boson",
                "conservation": "conservation", "symmetry": "symétrie", "invariance": "invariance",
                "quantum": "quantique", "classical": "classique", "relativistic": "relativiste",
                "detector": "détecteur", "calorimeter": "calorimètre", "tracker": "trajectographe",
                "collision": "collision", "interaction": "interaction", "decay": "désintégration",
                "cross_section": "section efficace", "luminosity": "luminosité", "efficiency": "efficacité"
            },
            "de": {
                "energy": "Energie", "momentum": "Impuls", "mass": "Masse", "velocity": "Geschwindigkeit",
                "particle": "Teilchen", "photon": "Photon", "electron": "Elektron", "proton": "Proton",
                "neutron": "Neutron", "quark": "Quark", "lepton": "Lepton", "boson": "Boson",
                "conservation": "Erhaltung", "symmetry": "Symmetrie", "invariance": "Invarianz",
                "quantum": "quantisch", "classical": "klassisch", "relativistic": "relativistisch",
                "detector": "Detektor", "calorimeter": "Kalorimeter", "tracker": "Spurendetektor",
                "collision": "Kollision", "interaction": "Wechselwirkung", "decay": "Zerfall",
                "cross_section": "Wirkungsquerschnitt", "luminosity": "Luminosität", "efficiency": "Effizienz"
            },
            "ja": {
                "energy": "エネルギー", "momentum": "運動量", "mass": "質量", "velocity": "速度",
                "particle": "粒子", "photon": "光子", "electron": "電子", "proton": "陽子",
                "neutron": "中性子", "quark": "クォーク", "lepton": "レプトン", "boson": "ボソン",
                "conservation": "保存", "symmetry": "対称性", "invariance": "不変性",
                "quantum": "量子", "classical": "古典", "relativistic": "相対論的",
                "detector": "検出器", "calorimeter": "カロリメーター", "tracker": "飛跡検出器",
                "collision": "衝突", "interaction": "相互作用", "decay": "崩壊",
                "cross_section": "断面積", "luminosity": "ルミノシティ", "efficiency": "効率"
            },
            "zh": {
                "energy": "能量", "momentum": "动量", "mass": "质量", "velocity": "速度",
                "particle": "粒子", "photon": "光子", "electron": "电子", "proton": "质子",
                "neutron": "中子", "quark": "夸克", "lepton": "轻子", "boson": "玻色子",
                "conservation": "守恒", "symmetry": "对称性", "invariance": "不变性",
                "quantum": "量子", "classical": "经典", "relativistic": "相对论",
                "detector": "探测器", "calorimeter": "量热器", "tracker": "径迹探测器",
                "collision": "碰撞", "interaction": "相互作用", "decay": "衰变",
                "cross_section": "截面", "luminosity": "亮度", "efficiency": "效率"
            }
        }
    
    def get_unit_translation(self, unit: str, language: str) -> str:
        """Get translation for a physics unit."""
        return self.unit_translations.get(language, {}).get(unit, unit)
    
    def get_physics_term(self, term: str, language: str) -> str:
        """Get translation for a physics term."""
        return self.physics_terms.get(language, {}).get(term, term)
    
    def format_physics_value(
        self, 
        value: float, 
        unit: str, 
        language: str = "en",
        use_scientific: bool = True
    ) -> str:
        """Format physics value with localized unit."""
        
        # Get locale info
        locale_info = SUPPORTED_LOCALES.get(language, SUPPORTED_LOCALES["en"])
        
        # Format number
        if use_scientific and abs(value) >= 1000 or abs(value) <= 0.001:
            if language in ["zh", "ja"]:
                # Use Unicode superscripts for CJK languages
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.2e}"
        else:
            # Regular formatting with locale-specific separators
            if locale_info.thousands_separator == ",":
                formatted_value = f"{value:,.2f}"
            else:
                formatted_value = f"{value:.2f}".replace(".", locale_info.decimal_separator)
        
        # Get unit translation
        unit_translation = self.get_unit_translation(unit, language)
        
        return f"{formatted_value} {unit_translation}"


# Supported locales configuration
SUPPORTED_LOCALES: Dict[str, LocaleInfo] = {
    "en": LocaleInfo(
        code="en",
        name="English",
        english_name="English",
        region="US",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y-%m-%d",
        time_format="%H:%M:%S",
        currency_symbol="$",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "es": LocaleInfo(
        code="es",
        name="Español",
        english_name="Spanish",
        region="ES",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S",
        currency_symbol="€",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "fr": LocaleInfo(
        code="fr",
        name="Français",
        english_name="French",
        region="FR",
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S",
        currency_symbol="€",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "de": LocaleInfo(
        code="de",
        name="Deutsch",
        english_name="German",
        region="DE",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        time_format="%H:%M:%S",
        currency_symbol="€",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "ja": LocaleInfo(
        code="ja",
        name="日本語",
        english_name="Japanese",
        region="JP",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y年%m月%d日",
        time_format="%H時%M分%S秒",
        currency_symbol="¥",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "zh": LocaleInfo(
        code="zh",
        name="中文",
        english_name="Chinese",
        region="CN",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y年%m月%d日",
        time_format="%H时%M分%S秒",
        currency_symbol="¥",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "ko": LocaleInfo(
        code="ko",
        name="한국어",
        english_name="Korean",
        region="KR",
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y년 %m월 %d일",
        time_format="%H시 %M분 %S초",
        currency_symbol="₩",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "it": LocaleInfo(
        code="it",
        name="Italiano",
        english_name="Italian",
        region="IT",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S",
        currency_symbol="€",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "pt": LocaleInfo(
        code="pt",
        name="Português",
        english_name="Portuguese",
        region="BR",
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S",
        currency_symbol="R$",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    ),
    "ru": LocaleInfo(
        code="ru",
        name="Русский",
        english_name="Russian",
        region="RU",
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d.%m.%Y",
        time_format="%H:%M:%S",
        currency_symbol="₽",
        scientific_notation=True,
        use_unicode_symbols=True,
        physics_unit_system="SI"
    )
}


class LocalizationManager:
    """
    Main localization manager for quantum task planning system.
    
    Provides translation, formatting, and physics-specific localization
    services across multiple languages and regions.
    """
    
    def __init__(
        self, 
        default_language: str = "en",
        translations_dir: str = "translations"
    ):
        self.default_language = default_language
        self.current_language = default_language
        self.translations_dir = Path(translations_dir)
        
        # Translation catalogs
        self.catalogs: Dict[str, gettext.GNUTranslations] = {}
        self.fallback_catalog: Optional[gettext.NullTranslations] = None
        
        # Physics translation manager
        self.physics_manager = PhysicsTranslationManager()
        
        # Message templates
        self.message_templates: Dict[str, Dict[str, str]] = {}
        
        # Initialize translation system
        self.initialize_translations()
        
        logger.info(f"Initialized localization manager with default language: {default_language}")
    
    def initialize_translations(self):
        """Initialize translation system and load catalogs."""
        
        # Create translations directory if it doesn't exist
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        
        # Load translation catalogs for each supported language
        for lang_code in SUPPORTED_LOCALES.keys():
            catalog_path = self.translations_dir / lang_code / "LC_MESSAGES"
            catalog_path.mkdir(parents=True, exist_ok=True)
            
            po_file = catalog_path / "messages.po"
            mo_file = catalog_path / "messages.mo"
            
            # Create default translation files if they don't exist
            if not po_file.exists():
                self._create_default_po_file(po_file, lang_code)
            
            # Load catalog if .mo file exists
            if mo_file.exists():
                try:
                    with open(mo_file, 'rb') as f:
                        catalog = gettext.GNUTranslations(f)
                        self.catalogs[lang_code] = catalog
                        logger.debug(f"Loaded translation catalog: {lang_code}")
                except Exception as e:
                    logger.error(f"Error loading catalog for {lang_code}: {e}")
        
        # Create fallback catalog
        self.fallback_catalog = gettext.NullTranslations()
        
        # Load message templates
        self.load_message_templates()
    
    def _create_default_po_file(self, po_file: Path, lang_code: str):
        """Create default .po file with common translations."""
        
        po_content = f'''# Quantum Task Planning Translations - {lang_code}
# Generated by DarkOperator Studio
#
msgid ""
msgstr ""
"Language: {lang_code}\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"

# Common interface terms
msgid "Task"
msgstr ""

msgid "Schedule"
msgstr ""

msgid "Execute"
msgstr ""

msgid "Status"
msgstr ""

msgid "Success"
msgstr ""

msgid "Failed"
msgstr ""

msgid "In Progress"
msgstr ""

msgid "Pending"
msgstr ""

# Physics terms
msgid "Energy"
msgstr ""

msgid "Momentum"
msgstr ""

msgid "Conservation"
msgstr ""

msgid "Quantum State"
msgstr ""

msgid "Optimization"
msgstr ""

msgid "Simulation"
msgstr ""

msgid "Anomaly Detection"
msgstr ""

msgid "Physics Validation"
msgstr ""

# Error messages
msgid "Task execution failed"
msgstr ""

msgid "Physics constraint violation"
msgstr ""

msgid "Security validation failed"
msgstr ""

msgid "Resource limit exceeded"
msgstr ""

# Status messages
msgid "Task completed successfully"
msgstr ""

msgid "Quantum schedule optimized"
msgstr ""

msgid "Physics constraints satisfied"
msgstr ""

msgid "System ready"
msgstr ""
'''
        
        with open(po_file, 'w', encoding='utf-8') as f:
            f.write(po_content)
        
        logger.info(f"Created default translation file: {po_file}")
    
    def load_message_templates(self):
        """Load message templates for different languages."""
        
        # Task execution messages
        self.message_templates = {
            "en": {
                "task_started": "Task '{task_name}' started with ID {task_id}",
                "task_completed": "Task '{task_name}' completed in {execution_time:.2f} seconds",
                "task_failed": "Task '{task_name}' failed: {error_message}",
                "physics_violation": "Physics constraint violation in task {task_id}: {violation_type}",
                "optimization_complete": "Quantum optimization completed: {algorithm} found solution with energy {energy:.6f}",
                "security_alert": "Security alert: {alert_type} detected in task {task_id}",
                "resource_usage": "Resource usage: CPU {cpu:.1f}%, Memory {memory:.1f} MB, GPU {gpu:.1f}%"
            },
            "es": {
                "task_started": "Tarea '{task_name}' iniciada con ID {task_id}",
                "task_completed": "Tarea '{task_name}' completada en {execution_time:.2f} segundos",
                "task_failed": "Tarea '{task_name}' falló: {error_message}",
                "physics_violation": "Violación de restricción física en tarea {task_id}: {violation_type}",
                "optimization_complete": "Optimización cuántica completada: {algorithm} encontró solución con energía {energy:.6f}",
                "security_alert": "Alerta de seguridad: {alert_type} detectado en tarea {task_id}",
                "resource_usage": "Uso de recursos: CPU {cpu:.1f}%, Memoria {memory:.1f} MB, GPU {gpu:.1f}%"
            },
            "fr": {
                "task_started": "Tâche '{task_name}' démarrée avec ID {task_id}",
                "task_completed": "Tâche '{task_name}' terminée en {execution_time:.2f} secondes",
                "task_failed": "Tâche '{task_name}' échouée : {error_message}",
                "physics_violation": "Violation de contrainte physique dans la tâche {task_id} : {violation_type}",
                "optimization_complete": "Optimisation quantique terminée : {algorithm} a trouvé une solution avec énergie {energy:.6f}",
                "security_alert": "Alerte de sécurité : {alert_type} détecté dans la tâche {task_id}",
                "resource_usage": "Utilisation des ressources : CPU {cpu:.1f}%, Mémoire {memory:.1f} MB, GPU {gpu:.1f}%"
            },
            "de": {
                "task_started": "Aufgabe '{task_name}' gestartet mit ID {task_id}",
                "task_completed": "Aufgabe '{task_name}' abgeschlossen in {execution_time:.2f} Sekunden",
                "task_failed": "Aufgabe '{task_name}' fehlgeschlagen: {error_message}",
                "physics_violation": "Physik-Constraint-Verletzung in Aufgabe {task_id}: {violation_type}",
                "optimization_complete": "Quantenoptimierung abgeschlossen: {algorithm} fand Lösung mit Energie {energy:.6f}",
                "security_alert": "Sicherheitsalarm: {alert_type} in Aufgabe {task_id} erkannt",
                "resource_usage": "Ressourcenverbrauch: CPU {cpu:.1f}%, Speicher {memory:.1f} MB, GPU {gpu:.1f}%"
            },
            "ja": {
                "task_started": "タスク '{task_name}' が ID {task_id} で開始されました",
                "task_completed": "タスク '{task_name}' が {execution_time:.2f} 秒で完了しました",
                "task_failed": "タスク '{task_name}' が失敗しました: {error_message}",
                "physics_violation": "タスク {task_id} で物理制約違反: {violation_type}",
                "optimization_complete": "量子最適化完了: {algorithm} がエネルギー {energy:.6f} の解を発見",
                "security_alert": "セキュリティアラート: タスク {task_id} で {alert_type} を検出",
                "resource_usage": "リソース使用量: CPU {cpu:.1f}%, メモリ {memory:.1f} MB, GPU {gpu:.1f}%"
            },
            "zh": {
                "task_started": "任务 '{task_name}' 已启动，ID 为 {task_id}",
                "task_completed": "任务 '{task_name}' 在 {execution_time:.2f} 秒内完成",
                "task_failed": "任务 '{task_name}' 失败: {error_message}",
                "physics_violation": "任务 {task_id} 中的物理约束违反: {violation_type}",
                "optimization_complete": "量子优化完成: {algorithm} 找到能量为 {energy:.6f} 的解",
                "security_alert": "安全警报: 在任务 {task_id} 中检测到 {alert_type}",
                "resource_usage": "资源使用: CPU {cpu:.1f}%, 内存 {memory:.1f} MB, GPU {gpu:.1f}%"
            }
        }
    
    def set_language(self, language: str) -> bool:
        """Set current language for translations."""
        
        if language not in SUPPORTED_LOCALES:
            logger.warning(f"Unsupported language: {language}")
            return False
        
        self.current_language = language
        logger.info(f"Language set to: {language}")
        return True
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        
        languages = []
        for code, locale_info in SUPPORTED_LOCALES.items():
            languages.append({
                'code': code,
                'name': locale_info.name,
                'english_name': locale_info.english_name,
                'region': locale_info.region,
                'rtl': locale_info.rtl
            })
        
        return languages
    
    def translate(self, message: str, language: Optional[str] = None) -> str:
        """Translate message to specified language."""
        
        target_language = language or self.current_language
        
        # Use gettext catalog if available
        if target_language in self.catalogs:
            try:
                return self.catalogs[target_language].gettext(message)
            except Exception as e:
                logger.error(f"Translation error for {target_language}: {e}")
        
        # Fallback to default language
        if target_language != self.default_language and self.default_language in self.catalogs:
            try:
                return self.catalogs[self.default_language].gettext(message)
            except Exception:
                pass
        
        # Return original message if no translation available
        return message
    
    def format_message(
        self, 
        template_key: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """Format localized message template with parameters."""
        
        target_language = language or self.current_language
        
        # Get template from language-specific templates
        templates = self.message_templates.get(target_language, 
                    self.message_templates.get(self.default_language, {}))
        
        template = templates.get(template_key, template_key)
        
        try:
            return template.format(**kwargs)
        except Exception as e:
            logger.error(f"Message formatting error: {e}")
            return template
    
    def format_datetime(
        self, 
        dt: datetime, 
        language: Optional[str] = None,
        include_time: bool = True
    ) -> str:
        """Format datetime according to locale preferences."""
        
        target_language = language or self.current_language
        locale_info = SUPPORTED_LOCALES.get(target_language, SUPPORTED_LOCALES[self.default_language])
        
        if include_time:
            format_string = f"{locale_info.date_format} {locale_info.time_format}"
        else:
            format_string = locale_info.date_format
        
        try:
            return dt.strftime(format_string)
        except Exception as e:
            logger.error(f"DateTime formatting error: {e}")
            return str(dt)
    
    def format_number(
        self, 
        number: Union[int, float], 
        language: Optional[str] = None,
        decimal_places: int = 2
    ) -> str:
        """Format number according to locale preferences."""
        
        target_language = language or self.current_language
        locale_info = SUPPORTED_LOCALES.get(target_language, SUPPORTED_LOCALES[self.default_language])
        
        try:
            # Format with specified decimal places
            if isinstance(number, float):
                formatted = f"{number:.{decimal_places}f}"
            else:
                formatted = str(number)
            
            # Apply locale-specific separators
            if decimal_places > 0 and isinstance(number, float):
                formatted = formatted.replace(".", locale_info.decimal_separator)
            
            # Add thousands separator for large numbers
            if abs(number) >= 1000:
                parts = formatted.split(locale_info.decimal_separator)
                integer_part = parts[0]
                
                # Add thousands separators
                if len(integer_part) > 3:
                    formatted_integer = ""
                    for i, digit in enumerate(reversed(integer_part)):
                        if i > 0 and i % 3 == 0:
                            formatted_integer = locale_info.thousands_separator + formatted_integer
                        formatted_integer = digit + formatted_integer
                    
                    if len(parts) > 1:
                        formatted = formatted_integer + locale_info.decimal_separator + parts[1]
                    else:
                        formatted = formatted_integer
            
            return formatted
            
        except Exception as e:
            logger.error(f"Number formatting error: {e}")
            return str(number)
    
    def format_physics_value(
        self, 
        value: float, 
        unit: str,
        language: Optional[str] = None,
        use_scientific: Optional[bool] = None
    ) -> str:
        """Format physics value with localized unit."""
        
        target_language = language or self.current_language
        
        if use_scientific is None:
            locale_info = SUPPORTED_LOCALES.get(target_language, SUPPORTED_LOCALES[self.default_language])
            use_scientific = locale_info.scientific_notation
        
        return self.physics_manager.format_physics_value(
            value, unit, target_language, use_scientific
        )
    
    def translate_physics_term(self, term: str, language: Optional[str] = None) -> str:
        """Translate physics-specific term."""
        
        target_language = language or self.current_language
        return self.physics_manager.get_physics_term(term, target_language)
    
    def get_locale_info(self, language: Optional[str] = None) -> LocaleInfo:
        """Get locale information for language."""
        
        target_language = language or self.current_language
        return SUPPORTED_LOCALES.get(target_language, SUPPORTED_LOCALES[self.default_language])
    
    def create_translation_file(self, language: str, overwrite: bool = False) -> bool:
        """Create translation file template for new language."""
        
        if language not in SUPPORTED_LOCALES:
            logger.error(f"Unsupported language: {language}")
            return False
        
        catalog_path = self.translations_dir / language / "LC_MESSAGES"
        catalog_path.mkdir(parents=True, exist_ok=True)
        
        po_file = catalog_path / "messages.po"
        
        if po_file.exists() and not overwrite:
            logger.warning(f"Translation file already exists: {po_file}")
            return False
        
        self._create_default_po_file(po_file, language)
        return True
    
    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate translation completeness across languages."""
        
        validation_results = {}
        
        # Get all message keys from default language
        default_templates = self.message_templates.get(self.default_language, {})
        default_keys = set(default_templates.keys())
        
        # Check each language for missing translations
        for lang_code in SUPPORTED_LOCALES.keys():
            if lang_code == self.default_language:
                continue
            
            lang_templates = self.message_templates.get(lang_code, {})
            lang_keys = set(lang_templates.keys())
            
            missing_keys = default_keys - lang_keys
            
            if missing_keys:
                validation_results[lang_code] = list(missing_keys)
        
        return validation_results
    
    def get_translation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get translation statistics for all languages."""
        
        stats = {}
        default_count = len(self.message_templates.get(self.default_language, {}))
        
        for lang_code, locale_info in SUPPORTED_LOCALES.items():
            lang_templates = self.message_templates.get(lang_code, {})
            translated_count = len(lang_templates)
            
            completion_rate = (translated_count / default_count * 100) if default_count > 0 else 0
            
            stats[lang_code] = {
                'language_name': locale_info.name,
                'english_name': locale_info.english_name,
                'total_messages': default_count,
                'translated_messages': translated_count,
                'completion_rate': completion_rate,
                'catalog_loaded': lang_code in self.catalogs
            }
        
        return stats