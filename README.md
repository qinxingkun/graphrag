
# 🍽️ Chinese Cooking Knowledge Graph Dataset

（中式烹饪知识图谱数据集）

## 1. 数据集简介

数据集主要由 **两个 CSV 文件** 组成：

```text
data/
├── nodes.csv           # 节点数据
└── relationships.csv   # 关系数据
```

### 1.1 nodes.csv（节点文件）

`nodes.csv` 用于描述图谱中的**实体节点**，每一行对应一个节点。

#### 主要字段说明

| 字段名         | 含义                                      |
| ----------- | --------------------------------------- |
| nodeId      | 节点唯一标识（全局唯一）                            |
| name        | 实体名称（如菜名、食材名、步骤名）                       |
| labels      | 节点类型（如 Recipe、Ingredient、CookingStep 等） |
| description | 实体描述                                    |
| category    | 分类信息（支持多分类，逗号分隔）                        |
| conceptType | 概念类型（如 实体、过程、工具 等）                      |
| difficulty  | 难度等级（用于 Recipe）                         |
| prepTime    | 准备时间                                    |
| cookTime    | 烹饪时间                                    |
| stepNumber  | 步骤序号（用于 CookingStep）                    |
| methods     | 烹饪方法                                    |
| tools       | 使用工具                                    |
| synonyms    | 同义词                                     |
| filePath    | 原始数据来源路径（可选）                            |

#### 主要节点类型（Labels）

* **Recipe**：菜谱
* **Ingredient**：食材
* **CookingStep**：烹饪步骤
* **CookingMethod**：烹饪方法
* **CookingTool**：烹饪工具
* **DifficultyLevel**：难度等级
* **Category / RecipeCategory**：分类节点
* **ConceptType**：概念类型
* **Root**：层次结构根节点
* **GraphStatistics / RelationshipStatistics**：统计节点（自动生成）

---

### 1.2 relationships.csv（关系文件）

`relationships.csv` 用于描述节点之间的**关系边**。

#### 主要字段说明

| 字段名              | 含义         |
| ---------------- | ---------- |
| relationshipId   | 关系唯一标识     |
| relationshipType | 关系类型编码     |
| startNodeId      | 起始节点 ID    |
| endNodeId        | 终止节点 ID    |
| amount           | 用量（用于食材关系） |
| unit             | 单位         |
| step_order       | 步骤顺序       |

#### 核心关系类型

| 关系类型                | 含义        |
| ------------------- | --------- |
| REQUIRES            | 菜谱 → 食材   |
| CONTAINS_STEP       | 菜谱 → 步骤   |
| NEXT_STEP           | 步骤 → 下一步骤 |
| BELONGS_TO          | 实体 → 上位概念 |
| BELONGS_TO_CATEGORY | 实体 → 分类   |
| HAS_CONCEPT_TYPE    | 实体 → 概念类型 |
| SIMILAR             | 相似实体      |
| USES_SAME_TOOL      | 使用相同工具    |
| USES_SAME_METHOD    | 使用相同方法    |
| DIFFICULTY_LEVEL    | 菜谱 → 难度   |

