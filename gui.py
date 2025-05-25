import os
import sys
import numpy as np
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QTextEdit, QLabel, QComboBox, QHBoxLayout, 
    QInputDialog, QListWidget, QSplitter, QListWidgetItem, QTabWidget,
    QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
from neo4j import GraphDatabase
from neo4j.graph import Path
import networkx as nx

# Загрузка переменных окружения
load_dotenv()

class Neo4jManager:
    """Класс для управления подключением к Neo4j"""
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        self.driver = None
        
    def connect(self):
        """Установка соединения с Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            return True
        except Exception as e:
            print(f"Ошибка подключения к Neo4j: {e}")
            return False
    
    def execute_query(self, query, parameters=None):
        """Выполнение запроса к Neo4j"""
        if not self.driver:
            raise ConnectionError("Соединение с Neo4j не установлено")
            
        with self.driver.session() as session:
            return session.run(query, parameters)
    
    def close(self):
        """Закрытие соединения"""
        if self.driver:
            self.driver.close()

class GraphDataProcessor:
    """Класс для обработки данных графа"""
    @staticmethod
    def get_node_color(labels):
        """Возвращает цвет узла на основе его типа"""
        color_map = {
            'User': (255, 100, 100, 220),     # Красный - Пользователи
            'Group': (100, 100, 255, 220),    # Синий - Группы
            'Role': (100, 255, 100, 220),     # Зеленый - Роли
            'Privilege': (255, 165, 0, 220),  # Оранжевый - Привилегии
            'Permission': (255, 0, 255, 220)  # Фиолетовый - Разрешения
        }
        return color_map.get(labels[0] if labels else None, (200, 200, 200, 220))
    
    @staticmethod
    def process_graph_data(records):
        """Преобразует записи Neo4j в граф NetworkX"""
        G = nx.DiGraph()
        node_data = {}

        for record in records:
            for val in record.values():
                if isinstance(val, Path):
                    GraphDataProcessor._process_path(val, G, node_data)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, "start_node"):
                            GraphDataProcessor._process_relationship(item, G, node_data)
                elif hasattr(val, "start_node"):
                    GraphDataProcessor._process_relationship(val, G, node_data)
                elif hasattr(val, "labels"):
                    GraphDataProcessor._process_node(val, node_data)

        for node_id, data in node_data.items():
            G.add_node(node_id, **data)

        return G

    @staticmethod
    def _process_node(node, node_data):
        """Обрабатывает узел Neo4j"""
        if node.id not in node_data:
            labels = list(node.labels)
            props = dict(node)
            display_name = props.get('id', props.get('cn', str(node.id)))
            node_data[node.id] = {
                'labels': labels,
                'properties': props,
                'display_name': display_name,
                'color': GraphDataProcessor.get_node_color(labels)
            }

    @staticmethod
    def _process_relationship(rel, G, node_data):
        """Обрабатывает связь Neo4j"""
        u = rel.start_node.id
        v = rel.end_node.id
        G.add_edge(u, v, type=rel.type, properties=dict(rel))
        for node in [rel.start_node, rel.end_node]:
            GraphDataProcessor._process_node(node, node_data)

    @staticmethod
    def _process_path(path, G, node_data):
        """Обрабатывает путь Neo4j"""
        for node in path.nodes:
            GraphDataProcessor._process_node(node, node_data)
        for rel in path.relationships:
            GraphDataProcessor._process_relationship(rel, G, node_data)

class GraphVisualizer:
    """Класс для визуализации графа"""
    def __init__(self, graph_widget):
        self.graph_widget = graph_widget
        self.graph_widget.setBackground('w')
        
    def draw_graph(self, graph):
        """Отрисовывает граф в виджете"""
        self.graph_widget.clear()
        
        if len(graph.nodes) == 0:
            return "Граф пуст. Нет данных для отображения."

        # Рассчитываем позиции узлов
        pos = nx.spring_layout(graph, k=0.3, iterations=50)
        
        # Подготавливаем данные для отрисовки
        nodes = np.array([pos[node] for node in graph.nodes()])
        edges = np.array([(list(graph.nodes()).index(u), list(graph.nodes()).index(v)) 
                         for u, v in graph.edges()])
        colors = [graph.nodes[node]['color'] for node in graph.nodes()]
        sizes = [50 if 'User' in graph.nodes[node]['labels'] else 45 for node in graph.nodes()]

        # Создаем графический элемент графа
        graph_item = pg.GraphItem()
        graph_item.setData(
            pos=nodes,
            adj=edges,
            size=sizes,
            symbolPen={'color': 'k', 'width': 3},
            symbolBrush=colors,
            pen={'width': 3, 'color': (100, 100, 100, 150)},
            pxMode=True
        )

        # Настраиваем область просмотра
        view = self.graph_widget.addViewBox()
        view.addItem(graph_item)
        view.setAspectLocked(True)

        # Добавляем подписи узлов
        self._add_node_labels(graph, pos, view)
        
        # Добавляем подписи связей
        self._add_edge_labels(graph, pos, view)
        
        # Возвращаем статистику по графу
        return self._get_graph_stats(graph)

    def _add_node_labels(self, graph, pos, view):
        """Добавляет подписи к узлам"""
        node_font = QFont('Arial', 12)
        node_font.setBold(True)
        
        for node in graph.nodes():
            x, y = pos[node]
            display_name = graph.nodes[node]['display_name']
            if len(display_name) > 15:
                display_name = display_name[:12] + "..."
            
            text = pg.TextItem(text=display_name, color=(20, 20, 20))
            text.setFont(node_font)
            text.setAnchor((0.5, 0.5))
            text.setPos(x, y)
            view.addItem(text)

    def _add_edge_labels(self, graph, pos, view):
        """Добавляет подписи к связям"""
        edge_font = QFont('Arial', 10)
        edge_font.setBold(True)
        
        for u, v, data in graph.edges(data=True):
            if 'type' in data:
                x = (pos[u][0] + pos[v][0]) / 2
                y = (pos[u][1] + pos[v][1]) / 2
                edge_text = pg.TextItem(text=data['type'], color=(70, 70, 70))
                edge_text.setFont(edge_font)
                edge_text.setAnchor((0.5, 0.5))
                edge_text.setPos(x, y)
                view.addItem(edge_text)

    def _get_graph_stats(self, graph):
        """Возвращает статистику по графу"""
        edge_stats = {}
        for u, v, data in graph.edges(data=True):
            if 'type' in data:
                edge_type = data['type']
                edge_stats[edge_type] = edge_stats.get(edge_type, 0) + 1
        
        stats = "Типы связей:\n" + "\n".join([f"{k}: {v}" for k, v in edge_stats.items()])
        stats += f"\n\nОтображено узлов: {len(graph.nodes)}, связей: {len(graph.edges)}"
        return stats

class NodeListManager:
    """Класс для управления списками узлов"""
    def __init__(self, tab_widget):
        self.tab_widget = tab_widget
        self.node_lists = {
            'User': QListWidget(),
            'Group': QListWidget(),
            'Role': QListWidget(),
            'Privilege': QListWidget(),
            'Permission': QListWidget(),
            'Other': QListWidget()
        }
        self._setup_tabs()

    def _setup_tabs(self):
        """Настраивает вкладки для разных типов узлов"""
        for label, widget in self.node_lists.items():
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.addWidget(QLabel(self._get_russian_label(label) + f" ({widget.count()})"))
            tab_layout.addWidget(widget)
            self.tab_widget.addTab(tab, self._get_russian_label(label))
            widget.itemDoubleClicked.connect(self.on_node_double_click)

    def _get_russian_label(self, label):
        """Возвращает русское название для типа узла"""
        labels_map = {
            'User': 'Пользователи',
            'Group': 'Группы',
            'Role': 'Роли',
            'Privilege': 'Привилегии',
            'Permission': 'Разрешения',
            'Other': 'Другие'
        }
        return labels_map.get(label, label)

    def load_nodes(self, records):
        """Загружает узлы в списки"""
        for widget in self.node_lists.values():
            widget.clear()
        
        for record in records:
            labels = record['labels']
            node_id = record.get('id', record.get('name', 'Неизвестно'))
            node_type = labels[0] if labels else 'Other'
            
            if node_type not in self.node_lists:
                node_type = 'Other'
            
            item = QListWidgetItem(node_id)
            item.setData(Qt.UserRole, node_type)
            self.node_lists[node_type].addItem(item)
        
        self._update_tab_titles()

    def _update_tab_titles(self):
        """Обновляет заголовки вкладок с количеством узлов"""
        for i in range(self.tab_widget.count()):
            label = self.tab_widget.tabText(i)
            eng_label = next(k for k, v in {
                'User': 'Пользователи',
                'Group': 'Группы',
                'Role': 'Роли',
                'Privilege': 'Привилегии',
                'Permission': 'Разрешения',
                'Other': 'Другие'
            }.items() if v == label)
            count = self.node_lists[eng_label].count()
            self.tab_widget.setTabText(i, f"{label} ({count})")

    def on_node_double_click(self, item):
        """Обработчик двойного клика по узлу"""
        node_id = item.text()
        node_type = item.data(Qt.UserRole)
        return f""
        MATCH (n:{node_type} {{id: "{node_id}"}})-[r]-(related)
        RETURN n, r, related
        ""

class QueryManager:
    """Класс для управления запросами"""
    PRESET_QUERIES = [
        ("Все узлы и связи (LIMIT 100)", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"),
        ("Кратчайший путь между узлами", "MATCH path = shortestPath((u1:$type1 {id: $node1})-[*]->(u2:$type2 {id: $node2})) RETURN path"),
        ("Поиск по имени узла", "MATCH (n) WHERE n.id = $search OR n.name = $search WITH n MATCH (n)-[r]-(related) RETURN n, r, related"),
        ("Пользователи с правами администратора", "MATCH (n:User)-[r]-(related) WHERE 'admin' IN n.uid RETURN n, r, related")
    ]

    @staticmethod
    def get_preset_queries():
        """Возвращает список предустановленных запросов"""
        return [q[0] for q in QueryManager.PRESET_QUERIES]

    @staticmethod
    def get_preset_query(index):
        """Возвращает текст запроса по индексу"""
        return QueryManager.PRESET_QUERIES[index][1]

    @staticmethod
    def process_query_with_inputs(query, parent_window):
        """Обрабатывает запрос, требующий ввода данных"""
        if "$search" in query:
            search_value, ok = QInputDialog.getText(parent_window, "Поиск узла", "Введите имя узла:")
            if ok and search_value:
                return query.replace("$search", f'"{search_value}"')
            return None

        if "shortestPath" in query and ("$node1" in query or "$type1" in query):
            node1, ok1 = QInputDialog.getText(parent_window, "Кратчайший путь", "Введите имя первого узла:")
            if not ok1 or not node1:
                return None
            
            type1, ok1 = QInputDialog.getItem(parent_window, "Тип узла", "Выберите тип первого узла:", 
                                           ["User", "Group", "Role", "Privilege", "Permission"], 0, False)
            if not ok1:
                return None
            
            node2, ok2 = QInputDialog.getText(parent_window, "Кратчайший путь", "Введите имя второго узла:")
            if not ok2 or not node2:
                return None
            
            type2, ok2 = QInputDialog.getItem(parent_window, "Тип узла", "Выберите тип второго узла:", 
                                           ["User", "Group", "Role", "Privilege", "Permission"], 0, False)
            if not ok2:
                return None
            
            return query.replace("$node1", f'"{node1}"') \
                       .replace("$type1", type1) \
                       .replace("$node2", f'"{node2}"') \
                       .replace("$type2", type2)

        return query

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPA Hound - Анализатор безопасности FreeIPA")
        self.setGeometry(100, 100, 1600, 900)
        
        # Инициализация компонентов
        self.db = Neo4jManager()
        if not self.db.connect():
            sys.exit(1)
            
        self._setup_ui()
        self._load_initial_data()

    def _setup_ui(self):
        """Настраивает пользовательский интерфейс"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Основная область
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        
        # Боковая панель
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        
        # Добавляем виджеты
        splitter.addWidget(main_area)
        splitter.addWidget(side_panel)
        splitter.setSizes([1200, 400])
        
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(splitter)

        # Настройка панели управления
        self._setup_control_panel(main_layout)
        
        # Настройка виджета графа
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('w')
        main_layout.addWidget(self.graph_widget, stretch=1)
        self.visualizer = GraphVisualizer(self.graph_widget)
        
        # Настройка области результатов
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        main_layout.addWidget(self.result_text)
        
        # Настройка боковой панели
        self._setup_legend(side_layout)
        self._setup_node_lists(side_layout)

    def _setup_control_panel(self, layout):
        """Настраивает панель управления запросами"""
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        self.query_input.setMinimumWidth(500)

        self.execute_button = QPushButton("Выполнить запрос")
        self.execute_button.clicked.connect(self.execute_query)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(QueryManager.get_preset_queries())
        self.preset_combo.currentIndexChanged.connect(self.load_preset_query)

        control_layout.addWidget(QLabel("Запрос:"))
        control_layout.addWidget(self.query_input)
        control_layout.addWidget(self.preset_combo)
        control_layout.addWidget(self.execute_button)
        layout.addWidget(control_panel)

    def _setup_legend(self, layout):
        """Настраивает легенду цветов"""
        legend = QGroupBox("Легенда цветов узлов")
        legend_layout = QFormLayout()
        legend.setLayout(legend_layout)
        
        colors = [
            ("Пользователи", QColor(255, 100, 100)),
            ("Группы", QColor(100, 100, 255)),
            ("Роли", QColor(100, 255, 100)),
            ("Привилегии", QColor(255, 165, 0)),
            ("Разрешения", QColor(255, 0, 255)),
            ("Другие", QColor(200, 200, 200))
        ]
        
        for label, color in colors:
            color_label = QLabel()
            color_label.setStyleSheet(f"background-color: {color.name()};")
            color_label.setFixedSize(20, 20)
            legend_layout.addRow(color_label, QLabel(label))
        
        layout.addWidget(legend)

    def _setup_node_lists(self, layout):
        """Настраивает списки узлов"""
        self.tab_widget = QTabWidget()
        self.node_list_manager = NodeListManager(self.tab_widget)
        layout.addWidget(self.tab_widget)

    def _load_initial_data(self):
        """Загружает начальные данные при запуске"""
        self.load_all_nodes()
        self.plot_graph()

    def load_all_nodes(self):
        """Загружает все узлы из базы данных"""
        query = """
        MATCH (n) 
        RETURN labels(n) as labels, n.id as id, n.name as name
        ORDER BY labels(n)[0], coalesce(n.id, n.name)
        """
        
        result = self.db.execute_query(query)
        self.node_list_manager.load_nodes(result)

    def load_preset_query(self, index):
        """Загружает предустановленный запрос"""
        self.query_input.setText(QueryManager.get_preset_query(index))

    def plot_graph(self, query=None):
        """Отображает граф"""
        if query is None:
            query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"

        try:
            result = self.db.execute_query(query)
            G = GraphDataProcessor.process_graph_data(result)
            
            stats = self.visualizer.draw_graph(G)
            self.result_text.append(stats)
            
        except Exception as e:
            self.result_text.append(f"Ошибка: {str(e)}")

    def execute_query(self):
        """Выполняет запрос"""
        query = self.query_input.text().strip()
        if not query:
            query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"

        # Обработка запросов, требующих ввода данных
        processed_query = QueryManager.process_query_with_inputs(query, self)
        if processed_query is None:
            return  # Пользователь отменил ввод
            
        self.result_text.clear()
        self.result_text.append(f"Выполнение запроса:\n{processed_query}")
        self.plot_graph(processed_query)

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        self.db.close()
        event.accept()

def setup_app_style(app):
    """Настраивает стиль приложения"""
    app.setStyle('Fusion')
    palette = app.palette()
    
    colors = {
        'Window': (255, 255, 255),
        'WindowText': Qt.black,
        'Base': (255, 255, 255),
        'AlternateBase': (240, 240, 240),
        'ToolTipBase': Qt.white,
        'ToolTipText': Qt.black,
        'Text': Qt.black,
        'Button': (240, 240, 240),
        'ButtonText': Qt.black,
        'BrightText': Qt.red,
        'Link': (42, 130, 218),
        'Highlight': (42, 130, 218),
        'HighlightedText': Qt.white
    }
    
    for role, color in colors.items():
        palette.setColor(getattr(palette, role), QColor(*color) if isinstance(color, tuple) else color)
    
    app.setPalette(palette)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_app_style(app)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
