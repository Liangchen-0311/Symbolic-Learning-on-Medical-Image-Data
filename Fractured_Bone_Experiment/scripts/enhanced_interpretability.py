"""
End-to-End Causal Chain Interpretability Report.

Explains the COMPLETE flow from formula → statistical encoding → feature selection
→ classifier weights → class prediction, answering:
  1. WHY does a formula detect a specific fracture type? (physical/medical reasoning)
  2. HOW do formulas combine to produce the final class prediction?
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.symbolic.fracture_operators import register_fracture_operators, FRACTURE_OPERATORS

register_fracture_operators(TENSOR_OPERATORS)

OPERATOR_ARITY = {}
for name, (func, arity, _) in TENSOR_OPERATORS.items():
    OPERATOR_ARITY[name] = arity

BINARY_OPS = {n for n, a in OPERATOR_ARITY.items() if a == 2}
UNARY_OPS = {n for n, a in OPERATOR_ARITY.items() if a == 1}
TERMINAL_PREFIX = 'I_'

STAT_NAMES_12 = ['mean', 'std', 'max', 'skewness', 'kurtosis',
                 'q10', 'q25', 'q50', 'q75', 'q90', 'ratio_above_mean', 'range']
STAT_NAMES_16 = STAT_NAMES_12 + ['iqr', 'cv', 'energy', 'entropy_approx']

REGION_NAMES_5 = ['global', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
REGION_NAMES_7 = REGION_NAMES_5 + ['top_half', 'bottom_half']

CLINICAL_DICT = {
    'edge_mag': ('边缘强度', '骨折线在X光片上表现为灰度突变，梯度幅值直接反映骨折线的存在与强度'),
    'edge_x': ('水平边缘', '检测垂直走向的骨折线（如纵行骨折），因为垂直线的水平梯度最大'),
    'edge_y': ('垂直边缘', '检测水平走向的骨折线（如横断骨折），因为水平线的垂直梯度最大'),
    'edge_xx': ('二阶水平边缘', '二阶导数在骨折线两侧产生正负跳变，可定位骨折线精确位置'),
    'edge_yy': ('二阶垂直边缘', '同上，垂直方向的二阶导数用于定位水平骨折线'),
    'line_h': ('水平线检测', '横断骨折线呈水平走向，水平线检测器直接匹配此模式'),
    'line_v': ('垂直线检测', '纵行骨折线呈垂直走向，垂直线检测器直接匹配此模式'),
    'line_45': ('45°线检测', '斜行骨折线沿45°方向延伸，此检测器匹配斜行/螺旋骨折'),
    'line_135': ('135°线检测', '斜行骨折线沿135°方向延伸，此检测器匹配斜行/螺旋骨折'),
    'edge_diag_45': ('45°对角边缘', '螺旋骨折线呈对角走向，45°方向梯度最大'),
    'edge_diag_135': ('135°对角边缘', '螺旋骨折线呈对角走向，135°方向梯度最大'),
    'cortical_cont': ('骨皮质连续性', '正常骨皮质连续光滑；骨折时皮质断裂，连续性被打破，此算子量化断裂程度'),
    'discont_map': ('不连续图', '骨折导致骨结构中断，在X光上表现为灰度不连续跳变，此算子突出这些间隙'),
    'displace_ind': ('移位指标', '骨折移位时两侧骨块错开，导致局部不对称，此算子检测这种错位'),
    'black_tophat': ('黑帽变换', '在亮骨区域中提取暗细线——骨折线在亮骨中呈暗线，黑帽恰好匹配此物理特征'),
    'white_tophat': ('白帽变换', '在暗背景中提取亮细结构——骨碎片/骨痂呈亮斑，白帽恰好匹配此特征'),
    'local_entropy': ('局部熵', '骨折破坏正常骨小梁有序结构，导致局部纹理紊乱，熵值升高'),
    'local_range': ('局部极差', '骨折线处灰度急剧变化，局部极差（最大值-最小值）显著增大'),
    'local_contrast': ('局部对比度', '骨折线两侧灰度差异大，局部对比度直接量化此差异'),
    'local_std_5x5': ('局部标准差', '骨折区域灰度分布更分散，标准差高于正常骨区域'),
    'bone_enhance': ('骨增强', '锐化皮质骨边缘，使骨折线更清晰可辨'),
    'threshold_bone': ('骨分割', '将骨骼从软组织中分离，减少软组织对骨折检测的干扰'),
    'soft_suppress': ('软组织抑制', '消除肌肉/脂肪等软组织信号，仅保留骨骼信息，减少假阳性'),
    'lr_asymmetry': ('左右不对称', '单侧骨折导致左右两侧灰度分布不对称，此算子量化这种差异'),
    'tb_asymmetry': ('上下不对称', '骨折移位导致近端/远端骨块位置不对称，此算子量化移位程度'),
    'ms_edge': ('多尺度边缘', '骨折线粗细不一（细裂隙到宽间隙），多尺度检测可覆盖不同粗细的骨折'),
    'blob_detect': ('斑点检测', '粉碎性骨折产生多个骨碎片，在X光上表现为亮斑点，斑点检测器直接匹配'),
    'gabor_0': ('0° Gabor', '水平方向纹理滤波，对横断骨折线（水平走向）响应最强'),
    'gabor_45': ('45° Gabor', '45°方向纹理滤波，对斜行/螺旋骨折线响应最强'),
    'gabor_90': ('90° Gabor', '垂直方向纹理滤波，对纵行骨折线响应最强'),
    'gabor_mag': ('Gabor幅值', '方向无关的纹理强度，量化整体方向性纹理的强弱'),
    'dog': ('高斯差分', '近似拉普拉斯金字塔，增强多尺度边缘，使骨折线在不同尺度下均可见'),
    'lbp_like': ('LBP纹理', '局部二值模式编码微观纹理，骨折区域纹理模式与健康骨不同'),
    'corner_harris': ('Harris角点', '骨折线交叉/分叉处产生角点，Harris检测器定位这些关键点'),
    'peak_location_y': ('峰值Y位置', '骨折线在图像中的纵向位置，不同位置对应不同骨骼部位'),
    'sqrt_abs': ('绝对值开方', '压缩极端值动态范围，避免少数异常像素主导统计量'),
    'log1p_abs': ('对数压缩', '压缩大值范围，使强信号和弱信号都能被统计编码捕获'),
    'sigmoid': ('Sigmoid', '将值映射到[0,1]，使特征具有概率含义'),
    'relu': ('ReLU', '抑制负值，保留正向信号（如骨折线增强后的正响应）'),
    'negate': ('取反', '反转明暗，使暗骨折线变为亮线便于后续检测'),
    'abs': ('绝对值', '消除正负号，使特征只反映强度不反映方向'),
    'pow2': ('平方', '增强强信号、抑制弱信号，突出骨折线响应'),
    'normalize': ('归一化', '标准化数值范围，使不同公式输出可比较'),
    'flip_h': ('水平翻转', '构造对称参考，用于左右差异分析'),
    'flip_v': ('垂直翻转', '构造对称参考，用于上下差异分析'),
    'multiply': ('相乘', '两个特征图逐像素相乘=逻辑AND，仅两者都强的区域才保留'),
    'subtract': ('相减', '两个特征图相减=差异检测，突出两者不同的区域'),
    'add': ('相加', '两个特征图相加=逻辑OR，任一特征强的区域都保留'),
    'dilate': ('膨胀', '扩展亮区域，使细骨折线变粗便于检测'),
    'erode': ('腐蚀', '收缩亮区域，消除细小噪声'),
    'opening': ('开运算', '先腐蚀后膨胀，去除小亮斑（噪声），保留大结构（骨折线）'),
    'closing': ('闭运算', '先膨胀后腐蚀，填充小暗洞，使断裂的骨折线重新连通'),
    'blur': ('模糊', '平滑噪声，使骨折线更连贯'),
    'blur_7x7': ('7x7模糊', '大范围平滑，提取宏观特征'),
    'tophat': ('顶帽变换', '提取比周围亮/暗的细结构'),
    'downsample_2x': ('2x下采样', '降低分辨率，提取粗粒度特征'),
    'downsample_4x': ('4x下采样', '进一步降低分辨率，提取超粗粒度特征'),
    'stride_pool_4': ('4步长池化', '降维聚合，减少计算量同时保留空间结构'),
    'patch_histogram_4x4': ('4x4直方图', '将图像分为4x4块统计分布，保留空间位置信息'),
    'global_avg_pool': ('全局均值池化', '整个特征图的平均值，反映整体强度水平'),
    'global_max_pool': ('全局最大池化', '整个特征图的最大值，反映最显著特征'),
    'global_min_pool': ('全局最小池化', '整个特征图的最小值，反映最暗区域'),
    'global_std_pool': ('全局标准差池化', '整体变异度，高值表示骨折区域灰度变化大'),
    'pool_top_half': ('上半池化', '聚合上半部分，对应骨骼近端'),
    'pool_bottom_half': ('下半池化', '聚合下半部分，对应骨骼远端'),
    'pool_left_half': ('左半池化', '聚合左半部分'),
    'pool_right_half': ('右半池化', '聚合右半部分'),
    'pool_center': ('中心池化', '聚合中心区域，对应骨干'),
    'pool_quad_tr': ('右上象限池化', '聚合右上象限'),
    'pool_thirds_mid': ('中三分之一池化', '聚合骨干中段'),
    'pool_surround': ('周边池化', '聚合周边区域，对应皮质骨'),
    'std_top_half': ('上半标准差', '近端变异度'),
    'std_bottom_half': ('下半标准差', '远端变异度'),
    'ratio_above_mean': ('超均值比', '高信号占比，骨折区域此值通常偏高'),
    'high_freq': ('高频提取', '提取高频成分（边缘/噪声），骨折线属于高频信号'),
    'I_NEG': ('反相X光', '反转灰度使骨骼变亮、背景变暗，便于检测骨骼中的暗骨折线'),
    'I_BONE': ('骨增强通道', '预处理的骨增强图像，骨骼边缘更锐利'),
    'I_EDGE_PRIOR': ('边缘先验通道', '预计算的梯度图，已突出边缘信息'),
    'I_GRAY': ('灰度通道', '原始灰度图，保留全部信息'),
    'I_H': ('色调通道', 'HSV色调，反映颜色类别'),
    'I_S': ('饱和度通道', 'HSV饱和度，反映颜色纯度'),
    'I_G': ('绿色通道', 'RGB绿色分量'),
    'I_R': ('红色通道', 'RGB红色分量'),
    'I_B': ('蓝色通道', 'RGB蓝色分量'),
    'I_r': ('HSV红色', 'HSV空间红色分量'),
    'I_g': ('HSV绿色', 'HSV空间绿色分量'),
    'I_BY': ('亮度通道', '明暗信息'),
    'I_RG': ('红绿差', '颜色差异通道'),
    'I_SOFT': ('软组织通道', '软组织信息'),
}

STAT_CLINICAL = {
    'mean': ('均值', '整体信号强度——骨折区域均值偏高（高密度）或偏低（低密度）'),
    'std': ('标准差', '灰度分散程度——骨折区域std通常更高（灰度变化剧烈）'),
    'max': ('最大值', '最亮像素——骨碎片/骨痂处max极高'),
    'skewness': ('偏度', '分布不对称性——骨折区域常呈正偏（少数极亮像素拉高均值）'),
    'kurtosis': ('峰度', '分布尖峭程度——骨折区域峰度更高（集中在少数灰度值）'),
    'q10': ('10%分位', '低灰度端——反映骨折线暗区'),
    'q25': ('25%分位', '下四分位——反映暗区分布'),
    'q50': ('中位数', '中心趋势——比均值更鲁棒'),
    'q75': ('75%分位', '上四分位——反映亮区分布'),
    'q90': ('90%分位', '高灰度端——反映骨碎片亮区'),
    'ratio_above_mean': ('超均值比', '高信号占比——骨折区域此值异常'),
    'range': ('极差', '最大值-最小值——骨折区域range更大'),
    'iqr': ('四分位距', 'Q75-Q25，中间50%数据的范围——反映核心灰度分布宽度'),
    'cv': ('变异系数', 'std/mean——消除均值影响后的变异度，骨折区域cv更高'),
    'energy': ('能量', '灰度平方的均值——高能量=高密度区域（骨碎片）'),
    'entropy_approx': ('近似熵', '灰度分布的不确定性——骨折区域熵更高（纹理紊乱）'),
}

REGION_CLINICAL = {
    'global': ('全局', '整张图像，反映整体骨折特征'),
    'top_left': ('左上', '近端-左侧区域'),
    'top_right': ('右上', '近端-右侧区域'),
    'bottom_left': ('左下', '远端-左侧区域'),
    'bottom_right': ('右下', '远端-右侧区域'),
    'top_half': ('上半', '骨骼近端区域'),
    'bottom_half': ('下半', '骨骼远端区域'),
}

FRACTURE_TYPE_PATTERNS = {
    'Comminuted': {
        'name_cn': '粉碎性骨折',
        'clinical_desc': '骨碎裂成3块以上，X光表现为多条骨折线+多个骨碎片',
        'key_ops': ['blob_detect', 'white_tophat', 'discont_map', 'ms_edge', 'local_entropy'],
        'physical_reason': '粉碎性骨折的X光特征：(1)多个骨碎片→blob_detect/white_tophat检测亮斑 '
                          '(2)多条骨折线→ms_edge检测多尺度边缘 (3)骨结构严重紊乱→local_entropy升高 '
                          '(4)多处骨皮质断裂→discont_map多处不连续',
        'key_stats': ['entropy_approx', 'std', 'range', 'kurtosis'],
        'key_regions': ['global', 'top_half', 'bottom_half'],
    },
    'Greenstick': {
        'name_cn': '青枝骨折',
        'clinical_desc': '骨皮质一侧断裂、另一侧弯曲，类似折断青枝',
        'key_ops': ['cortical_cont', 'bone_enhance', 'lr_asymmetry', 'local_range'],
        'physical_reason': '青枝骨折的X光特征：(1)皮质一侧断裂→cortical_cont检测局部皮质中断 '
                          '(2)骨弯曲变形→lr_asymmetry检测左右不对称 (3)断裂处灰度突变→local_range增大 '
                          '(4)需骨增强看清→bone_enhance锐化皮质边缘',
        'key_stats': ['skewness', 'ratio_above_mean', 'std'],
        'key_regions': ['top_left', 'top_right', 'bottom_left', 'bottom_right'],
    },
    'Healthy': {
        'name_cn': '健康',
        'clinical_desc': '骨骼完整无骨折，皮质连续光滑',
        'key_ops': ['cortical_cont', 'soft_suppress', 'threshold_bone'],
        'physical_reason': '健康骨的X光特征：(1)皮质连续→cortical_cont值高且均匀 '
                          '(2)无骨折线→soft_suppress后无异常暗线 (3)骨轮廓完整→threshold_bone后形状规则 '
                          '健康样本的特征是"缺少异常"而非"存在异常"',
        'key_stats': ['std', 'entropy_approx', 'range'],
        'key_regions': ['global'],
    },
    'Linear': {
        'name_cn': '线形骨折',
        'clinical_desc': '细直线状骨折线，无移位',
        'key_ops': ['line_h', 'line_v', 'edge_mag', 'black_tophat', 'discont_map'],
        'physical_reason': '线形骨折的X光特征：(1)细直线→line_h/line_v直接匹配线状模式 '
                          '(2)骨折线细→black_tophat提取暗细线 (3)灰度突变→edge_mag高响应 '
                          '(4)骨皮质中断→discont_map显示不连续',
        'key_stats': ['max', 'range', 'q90'],
        'key_regions': ['top_half', 'bottom_half', 'center'],
    },
    'Oblique Displaced': {
        'name_cn': '斜行移位骨折',
        'clinical_desc': '斜行骨折线+骨块移位',
        'key_ops': ['edge_diag_45', 'edge_diag_135', 'displace_ind', 'tb_asymmetry', 'line_45', 'line_135'],
        'physical_reason': '斜行移位骨折的X光特征：(1)斜行骨折线→edge_diag_45/135检测对角边缘 '
                          '(2)骨块移位→displace_ind检测错位 (3)上下不对称→tb_asymmetry量化移位程度 '
                          '(4)斜线模式→line_45/line_135匹配斜行方向',
        'key_stats': ['skewness', 'mean', 'ratio_above_mean'],
        'key_regions': ['top_half', 'bottom_half'],
    },
    'Oblique': {
        'name_cn': '斜行骨折',
        'clinical_desc': '斜行骨折线，无移位',
        'key_ops': ['edge_diag_45', 'edge_diag_135', 'line_45', 'line_135', 'gabor_45'],
        'physical_reason': '斜行骨折的X光特征：(1)斜行骨折线→edge_diag_45/135检测对角方向梯度 '
                          '(2)斜线模式→line_45/line_135匹配 (3)方向性纹理→gabor_45对45°方向响应最强 '
                          '与"斜行移位"的区别是无移位指标和不对称指标',
        'key_stats': ['max', 'std', 'range'],
        'key_regions': ['global', 'top_half'],
    },
    'Segmental': {
        'name_cn': '节段性骨折',
        'clinical_desc': '骨在两处以上断裂，形成游离骨段',
        'key_ops': ['discont_map', 'ms_edge', 'local_entropy', 'tb_asymmetry', 'patch_histogram_4x4'],
        'physical_reason': '节段性骨折的X光特征：(1)多处断裂→discont_map多处不连续 '
                          '(2)多尺度边缘→ms_edge覆盖不同粗细的骨折线 (3)结构严重紊乱→local_entropy高 '
                          '(4)空间分布不均→patch_histogram_4x4捕获多峰分布',
        'key_stats': ['entropy_approx', 'kurtosis', 'iqr'],
        'key_regions': ['global', 'top_half', 'bottom_half'],
    },
    'Spiral': {
        'name_cn': '螺旋骨折',
        'clinical_desc': '骨折线呈螺旋状环绕骨干',
        'key_ops': ['edge_diag_45', 'edge_diag_135', 'line_45', 'line_135', 'gabor_45', 'gabor_mag'],
        'physical_reason': '螺旋骨折的X光特征：(1)螺旋线在对角方向延伸→edge_diag_45/135检测 '
                          '(2)斜线成分→line_45/line_135匹配 (3)方向性纹理→gabor_45响应强 '
                          '(4)整体方向性→gabor_mag量化方向纹理强度 '
                          '螺旋骨折的关键特征是多方向对角边缘同时存在',
        'key_stats': ['std', 'range', 'entropy_approx'],
        'key_regions': ['global', 'center'],
    },
    'Transverse Displaced': {
        'name_cn': '横断移位骨折',
        'clinical_desc': '水平骨折线+骨块移位',
        'key_ops': ['line_h', 'edge_mag', 'displace_ind', 'tb_asymmetry', 'lr_asymmetry'],
        'physical_reason': '横断移位骨折的X光特征：(1)水平骨折线→line_h直接匹配 '
                          '(2)骨折线梯度→edge_mag高响应 (3)骨块移位→displace_ind检测错位 '
                          '(4)不对称→tb_asymmetry(上下)+lr_asymmetry(左右)量化移位',
        'key_stats': ['skewness', 'mean', 'ratio_above_mean'],
        'key_regions': ['top_half', 'bottom_half'],
    },
    'Transverse': {
        'name_cn': '横断骨折',
        'clinical_desc': '水平骨折线，无移位',
        'key_ops': ['line_h', 'edge_mag', 'black_tophat', 'gabor_0', 'edge_y'],
        'physical_reason': '横断骨折的X光特征：(1)水平骨折线→line_h直接匹配水平线模式 '
                          '(2)骨折线梯度→edge_mag高响应 (3)暗细线→black_tophat提取 '
                          '(4)水平纹理→gabor_0对0°方向响应最强 (5)垂直梯度→edge_y检测水平边缘',
        'key_stats': ['max', 'std', 'q90'],
        'key_regions': ['global', 'center'],
    },
}


class FormulaParser:
    def __init__(self, formula_str):
        self.tokens = formula_str.strip().split()
        self.tree = self._build_tree()

    def _build_tree(self):
        stack = []
        for tok in self.tokens:
            if tok.startswith(TERMINAL_PREFIX):
                stack.append({'type': 'terminal', 'name': tok, 'children': []})
            elif tok in OPERATOR_ARITY:
                arity = OPERATOR_ARITY[tok]
                if len(stack) < arity:
                    return None
                children = [stack.pop() for _ in range(arity)]
                children.reverse()
                stack.append({'type': 'operator', 'name': tok, 'children': children})
            else:
                return None
        if len(stack) != 1:
            return None
        return stack[0]

    def get_operator_set(self, node=None):
        if node is None:
            node = self.tree
        if node is None:
            return set()
        ops = set()
        if node['type'] == 'operator':
            ops.add(node['name'])
            for child in node['children']:
                ops |= self.get_operator_set(child)
        return ops

    def get_terminal_set(self, node=None):
        if node is None:
            node = self.tree
        if node is None:
            return set()
        terms = set()
        if node['type'] == 'terminal':
            terms.add(node['name'])
        else:
            for child in node['children']:
                terms |= self.get_terminal_set(child)
        return terms

    def to_clinical_chain(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node is None:
            return []
        chain = []
        indent = '    ' * depth
        if node['type'] == 'terminal':
            cn, desc = CLINICAL_DICT.get(node['name'], (node['name'], '未知输入'))
            chain.append(f'{indent}[IN] {node["name"]} -- {cn}')
            chain.append(f'{indent}     物理含义: {desc}')
            return chain
        name = node['name']
        cn, desc = CLINICAL_DICT.get(name, (name, '未知操作'))
        if name in BINARY_OPS:
            chain.append(f'{indent}[BIN] {name}({cn})')
            chain.append(f'{indent}     作用: {desc}')
            for child in node['children']:
                chain.extend(self.to_clinical_chain(child, depth + 1))
        elif name in UNARY_OPS:
            chain.append(f'{indent}[UNI] {name}({cn})')
            chain.append(f'{indent}     作用: {desc}')
            chain.extend(self.to_clinical_chain(node['children'][0], depth + 1))
        else:
            chain.append(f'{indent}[???] {name}')
        return chain


def load_classifier_and_trace(output_dir, formulas, stats_per_formula):
    output_dir = Path(output_dir)
    classifier_path = output_dir / 'best_classifier.pkl'
    if not classifier_path.exists():
        return None

    import joblib
    clf_data = joblib.load(classifier_path)
    pipe = clf_data['pipe']
    method = clf_data.get('method', 'unknown')
    anova_selector = clf_data.get('anova_selector')
    mi_selector = clf_data.get('mi_selector')
    non_const_mask = clf_data.get('non_const')

    from sklearn.ensemble import StackingClassifier, VotingClassifier

    if isinstance(pipe, StackingClassifier):
        inner_names = [name for name, _ in pipe.estimators]
        if hasattr(pipe, 'estimators_') and pipe.estimators_ is not None:
            fitted_inner = dict(zip(inner_names, pipe.estimators_))
        else:
            fitted_inner = {name: est for name, est in pipe.estimators}
        primary_key = None
        for k in ['hgb_mi_sw', 'hgb_mi', 'hgb_sw', 'hgb']:
            if k in fitted_inner:
                primary_key = k
                break
        if primary_key is None:
            primary_key = list(fitted_inner.keys())[0]
        primary_pipe = fitted_inner[primary_key]
        clf = primary_pipe.named_steps['clf']
        selector = primary_pipe.named_steps.get('select', None)
    elif isinstance(pipe, VotingClassifier):
        inner_names = [name for name, _ in pipe.estimators]
        if hasattr(pipe, 'estimators_') and pipe.estimators_ is not None:
            fitted_inner = dict(zip(inner_names, pipe.estimators_))
        else:
            fitted_inner = {name: est for name, est in pipe.estimators}
        primary_key = list(fitted_inner.keys())[0]
        inner_pipe = fitted_inner[primary_key]
        clf = inner_pipe.named_steps['clf']
        selector = inner_pipe.named_steps.get('select', None)
    elif hasattr(pipe, 'named_steps'):
        primary_key = method
        clf = pipe.named_steps['clf']
        selector = pipe.named_steps.get('select', None)
    else:
        return None

    selected_idx = None
    original_idx = None
    importances = None
    imp_label = None

    if selector is not None and hasattr(selector, 'get_support'):
        try:
            selected_idx = selector.get_support(indices=True)
        except Exception:
            selected_idx = None

        if selected_idx is not None:
            use_mi = method in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
            mi_pool_sel = clf_data.get('mi_pool_selector')
            if use_mi and mi_pool_sel is not None and mi_selector is not None:
                try:
                    mi_pool_mask = mi_pool_sel.get_support(indices=True)
                    mi_within_pool = mi_selector.get_support(indices=True)
                    mi_to_pool = mi_pool_mask[mi_within_pool]
                    original_idx = mi_to_pool[selected_idx]
                    if non_const_mask is not None:
                        original_idx = np.where(non_const_mask)[0][original_idx]
                except Exception:
                    if anova_selector is not None:
                        anova_mask = anova_selector.get_support(indices=True)
                        original_idx = anova_mask[selected_idx]
                        if non_const_mask is not None:
                            original_idx = np.where(non_const_mask)[0][original_idx]
                    else:
                        original_idx = selected_idx
            elif anova_selector is not None:
                anova_mask = anova_selector.get_support(indices=True)
                original_idx = anova_mask[selected_idx]
                if non_const_mask is not None:
                    original_idx = np.where(non_const_mask)[0][original_idx]
            else:
                original_idx = selected_idx

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        imp_label = 'importance'
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_)
        imp_label = 'weight'
    else:
        importances = None

    return {
        'method': method,
        'primary_key': primary_key,
        'clf': clf,
        'selector': selector,
        'selected_idx': selected_idx,
        'original_idx': original_idx,
        'importances': importances,
        'imp_label': imp_label,
        'n_selected': len(selected_idx) if selected_idx is not None else 0,
    }


def generate_report(output_dir):
    output_dir = Path(output_dir)
    validated_path = output_dir / 'validated_formulas.json'
    if not validated_path.exists():
        print(f"Error: {validated_path} not found")
        return

    formulas = json.load(open(validated_path))
    formulas.sort(key=lambda f: f.get('full_res_accuracy', f.get('accuracy', 0)), reverse=True)

    class_names_path = output_dir / 'class_names.json'
    class_names = json.load(open(class_names_path)) if class_names_path.exists() else [
        'Comminuted', 'Greenstick', 'Healthy', 'Linear',
        'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
        'Transverse Displaced', 'Transverse',
    ]

    results_path = output_dir / 'classifier_results.json'
    classifier_results = json.load(open(results_path)) if results_path.exists() else {}
    per_class = classifier_results.get('per_class_test', {})

    dist_cfg = {}
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'fracture_v3_expanded.yaml'
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        dist_cfg = cfg.get('phase3', {}).get('distribution_stats', {})

    n_stats = dist_cfg.get('n_stats', 16)
    n_regions = dist_cfg.get('n_regions', 7)
    stats_per_formula = n_stats * n_regions

    stat_names = STAT_NAMES_16 if n_stats >= 16 else STAT_NAMES_12
    region_names = REGION_NAMES_7 if n_regions >= 7 else REGION_NAMES_5

    clf_info = load_classifier_and_trace(output_dir, formulas, stats_per_formula)

    lines = []
    lines.append('=' * 90)
    lines.append('  骨折符号特征端到端因果链可解释性报告')
    lines.append('  End-to-End Causal Chain Interpretability Report')
    lines.append('=' * 90)
    lines.append('')

    # ===== Part 1: Pipeline Overview =====
    lines.append('=' * 90)
    lines.append('  第一部分: 完整预测流程概览')
    lines.append('=' * 90)
    lines.append('')
    lines.append('  从X光图像到骨折类型预测，经过以下5个阶段:')
    lines.append('')
    lines.append('  阶段1 [公式执行]:')
    lines.append('    X光图像 --> 输入通道(I_BONE/I_NEG/I_EDGE_PRIOR/...) --> 算子组合 --> 特征图[B,H,W]')
    lines.append('    每个公式将图像变换为一个2D特征图，突出某种骨折相关的视觉模式')
    lines.append('')
    lines.append('  阶段2 [统计编码]:')
    lines.append(f'    特征图[B,H,W] --> {n_stats}个统计量 x {n_regions}个空间区域 = {stats_per_formula}维特征向量')
    lines.append(f'    统计量: {", ".join(stat_names)}')
    lines.append(f'    空间区域: {", ".join(region_names)}')
    lines.append('    每个公式贡献一组统计特征，将2D空间信息压缩为1D数值特征')
    lines.append('')
    lines.append('  阶段3 [特征选择]:')
    if clf_info and clf_info.get('method'):
        lines.append(f'    方法: {clf_info["method"]}')
        lines.append(f'    从 {len(formulas)} x {stats_per_formula} = {len(formulas)*stats_per_formula} 维')
        lines.append(f'    --> 选出 {clf_info["n_selected"]} 维关键特征')
    else:
        lines.append(f'    从 {len(formulas)} x {stats_per_formula} = {len(formulas)*stats_per_formula} 维中选出关键特征')
    lines.append('    ANOVA: 线性关系筛选 | MI: 非线性关系筛选')
    lines.append('')
    lines.append('  阶段4 [分类器训练]:')
    if clf_info:
        lines.append(f'    最佳模型: {clf_info["method"]} (子模型: {clf_info.get("primary_key", "N/A")})')
    lines.append('    HistGradientBoosting: 基于决策树集成，每个树节点按特征阈值分裂')
    lines.append('')
    lines.append('  阶段5 [类别预测]:')
    lines.append('    输入特征 --> 决策树序列投票 --> 10类骨折概率 --> argmax --> 最终预测')
    lines.append('')

    # ===== Part 2: Per-class explanation =====
    lines.append('=' * 90)
    lines.append('  第二部分: 各骨折类型的因果诊断链')
    lines.append('  (为什么这些算子能检测这种骨折? 算子如何叠合生成预测?)')
    lines.append('=' * 90)
    lines.append('')

    for ftype, info in FRACTURE_TYPE_PATTERNS.items():
        acc = per_class.get(ftype, 0)
        lines.append('-' * 90)
        lines.append(f'  [{info["name_cn"]}] ({ftype})  测试准确率: {acc:.1%}')
        lines.append('-' * 90)
        lines.append('')
        lines.append(f'  临床描述: {info["clinical_desc"]}')
        lines.append('')
        lines.append(f'  物理因果链 (为什么这些算子有效):')
        for sentence in info['physical_reason'].split('(1)')[1:]:
            sentence = sentence.strip()
            if sentence:
                lines.append(f'    ({sentence[0]}){sentence[1:]}')

        lines.append('')
        lines.append(f'  关键算子 -> 物理映射:')
        for op in info['key_ops']:
            cn, desc = CLINICAL_DICT.get(op, (op, '未知'))
            lines.append(f'    {op}({cn}):')
            lines.append(f'      {desc}')
        lines.append('')

        lines.append(f'  关键统计量 -> 诊断含义:')
        for stat in info['key_stats']:
            if stat in STAT_CLINICAL:
                cn, desc = STAT_CLINICAL[stat]
                lines.append(f'    {stat}({cn}): {desc}')
        lines.append('')

        lines.append(f'  关键空间区域 -> 解剖含义:')
        for reg in info['key_regions']:
            if reg in REGION_CLINICAL:
                cn, desc = REGION_CLINICAL[reg]
                lines.append(f'    {reg}({cn}): {desc}')
        lines.append('')

        matched_formulas = []
        for f in formulas[:100]:
            p = FormulaParser(f['str'])
            ops = p.get_operator_set()
            overlap = ops & set(info['key_ops'])
            if len(overlap) >= 2:
                acc_f = f.get('full_res_accuracy', f.get('accuracy', 0))
                matched_formulas.append((acc_f, f['str'], sorted(overlap)))
        matched_formulas.sort(key=lambda x: -x[0])

        if matched_formulas:
            lines.append(f'  匹配的公式 (>=2个关键算子, TOP-5):')
            for rank, (acc_f, fstr, overlap) in enumerate(matched_formulas[:5]):
                lines.append(f'    公式{rank+1}: acc={acc_f:.3f}')
                lines.append(f'      表达式: {fstr[:70]}{"..." if len(fstr)>70 else ""}')
                lines.append(f'      匹配算子: {", ".join(overlap)}')
                lines.append(f'      叠合过程:')
                p = FormulaParser(fstr)
                chain = p.to_clinical_chain()
                for c in chain[:8]:
                    lines.append(f'        {c}')
                if len(chain) > 8:
                    lines.append(f'        ... (共{len(chain)}步)')
                lines.append(f'      --> 输出特征图 --> 统计编码({n_stats}x{n_regions}={stats_per_formula}维)')
                lines.append(f'      --> 特征选择 --> 分类器投票 --> 贡献到[{info["name_cn"]}]类别')
                lines.append('')
        lines.append('')

    # ===== Part 3: Feature tracing from formula to class =====
    lines.append('=' * 90)
    lines.append('  第三部分: 从公式到类别的特征追踪')
    lines.append('  (哪些公式的哪些统计量被分类器选中，对哪个类别影响最大?)')
    lines.append('=' * 90)
    lines.append('')

    if clf_info and clf_info.get('original_idx') is not None and clf_info.get('importances') is not None:
        original_idx = clf_info['original_idx']
        importances = clf_info['importances']

        formula_contributions = defaultdict(lambda: defaultdict(float))
        formula_stat_details = defaultdict(list)

        if importances.ndim == 2:
            for fi in range(len(original_idx)):
                orig = original_idx[fi]
                formula_idx = orig // stats_per_formula
                stat_idx = orig % stats_per_formula
                sname = stat_names[stat_idx] if stat_idx < len(stat_names) else f'stat_{stat_idx}'
                rname = region_names[stat_idx // n_stats] if (stat_idx // n_stats) < len(region_names) else f'region_{stat_idx // n_stats}'
                actual_stat_name = stat_names[stat_idx % n_stats] if (stat_idx % n_stats) < len(stat_names) else f's_{stat_idx%n_stats}'

                for class_idx in range(importances.shape[0]):
                    imp = importances[class_idx, fi]
                    if imp > 0:
                        cname = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
                        formula_contributions[formula_idx][cname] += imp
                        if class_idx == 0 or True:
                            formula_stat_details[formula_idx].append({
                                'stat': actual_stat_name,
                                'region': rname,
                                'class': class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}',
                                'importance': float(imp),
                            })

            lines.append(f'  分类器: {clf_info["method"]} (子模型: {clf_info.get("primary_key", "N/A")})')
            lines.append(f'  选中特征数: {len(original_idx)}')
            lines.append('')

            for fidx in sorted(formula_contributions.keys()):
                if fidx >= len(formulas):
                    continue
                fstr = formulas[fidx]['str']
                acc = formulas[fidx].get('full_res_accuracy', formulas[fidx].get('accuracy', 0))
                class_imps = formula_contributions[fidx]

                lines.append(f'  公式[{fidx}] acc={acc:.3f}')
                lines.append(f'    表达式: {fstr[:80]}{"..." if len(fstr)>80 else ""}')

                p = FormulaParser(fstr)
                key_ops_found = p.get_operator_set() & set(FRACTURE_OPERATORS.keys())
                if key_ops_found:
                    ops_str = ', '.join(f'{op}({CLINICAL_DICT.get(op, (op, ""))[0]})' for op in sorted(key_ops_found))
                    lines.append(f'    骨折专用算子: {ops_str}')

                sorted_classes = sorted(class_imps.items(), key=lambda x: -x[1])[:5]
                lines.append(f'    对各类别的贡献 (特征重要性累加):')
                for cname, imp_sum in sorted_classes:
                    bar = '#' * int(imp_sum * 200)
                    lines.append(f'      {cname:25s}: {imp_sum:.4f} {bar}')

                top_stats = sorted(formula_stat_details[fidx], key=lambda x: -x['importance'])[:5]
                if top_stats:
                    lines.append(f'    最关键的统计特征:')
                    for sd in top_stats:
                        lines.append(f'      {sd["region"]}.{sd["stat"]} -> {sd["class"]}: importance={sd["importance"]:.4f}')
                lines.append('')
        else:
            lines.append(f'  分类器: {clf_info["method"]}')
            lines.append(f'  选中特征数: {len(original_idx)}')

            formula_imp = defaultdict(float)
            formula_top_stats = defaultdict(list)

            for fi in range(len(original_idx)):
                orig = original_idx[fi]
                formula_idx = orig // stats_per_formula
                stat_idx = orig % stats_per_formula
                actual_stat_name = stat_names[stat_idx % n_stats] if (stat_idx % n_stats) < len(stat_names) else f's_{stat_idx%n_stats}'
                rname = region_names[stat_idx // n_stats] if (stat_idx // n_stats) < len(region_names) else f'region_{stat_idx // n_stats}'

                imp = importances[fi]
                formula_imp[formula_idx] += imp
                formula_top_stats[formula_idx].append({
                    'stat': actual_stat_name,
                    'region': rname,
                    'importance': float(imp),
                })

            sorted_formulas = sorted(formula_imp.items(), key=lambda x: -x[1])[:20]

            lines.append(f'  TOP-20 公式对分类器的贡献:')
            lines.append('')

            for rank, (fidx, total_imp) in enumerate(sorted_formulas):
                if fidx >= len(formulas):
                    continue
                fstr = formulas[fidx]['str']
                acc = formulas[fidx].get('full_res_accuracy', formulas[fidx].get('accuracy', 0))

                lines.append(f'  #{rank+1} 公式[{fidx}] 总重要性={total_imp:.4f} acc={acc:.3f}')
                lines.append(f'    表达式: {fstr[:80]}{"..." if len(fstr)>80 else ""}')

                p = FormulaParser(fstr)
                key_ops_found = p.get_operator_set() & set(FRACTURE_OPERATORS.keys())
                if key_ops_found:
                    ops_str = ', '.join(f'{op}({CLINICAL_DICT.get(op, (op, ""))[0]})' for op in sorted(key_ops_found))
                    lines.append(f'    骨折专用算子: {ops_str}')

                top_stats = sorted(formula_top_stats[fidx], key=lambda x: -x['importance'])[:5]
                lines.append(f'    最关键的统计特征:')
                for sd in top_stats:
                    stat_cn = STAT_CLINICAL.get(sd['stat'], (sd['stat'], ''))[0]
                    region_cn = REGION_CLINICAL.get(sd['region'], (sd['region'], ''))[0]
                    lines.append(f'      {sd["region"]}({region_cn}).{sd["stat"]}({stat_cn}): importance={sd["importance"]:.4f}')

                lines.append(f'    叠合到预测的路径:')
                lines.append(f'      公式执行 --> 特征图 --> {sd["region"]}区域{sd["stat"]}统计量')
                lines.append(f'      --> 特征选择(被选中, importance={total_imp:.4f})')
                lines.append(f'      --> HGB决策树分裂节点 --> 投票决定类别')
                lines.append('')
    else:
        lines.append('  (无法加载分类器模型，跳过特征追踪)')

    # ===== Part 4: How formulas combine for prediction =====
    lines.append('=' * 90)
    lines.append('  第四部分: 公式如何叠合生成最终预测')
    lines.append('=' * 90)
    lines.append('')
    lines.append('  HistGradientBoosting分类器的工作原理:')
    lines.append('')
    lines.append('  1. 每棵决策树是一个if-then-else规则链:')
    lines.append('     例如: if formula[619].range > 0.5 then')
    lines.append('             if formula[481].median < 0.3 then')
    lines.append('               预测 = "Comminuted" (概率0.8)')
    lines.append('             else')
    lines.append('               预测 = "Linear" (概率0.6)')
    lines.append('')
    lines.append('  2. 多棵树投票(加权平均)得到最终概率:')
    lines.append('     P(Comminuted) = 0.3*tree1 + 0.2*tree2 + 0.15*tree3 + ...')
    lines.append('')
    lines.append('  3. 不同公式从不同角度描述骨折特征:')
    lines.append('     公式A: 检测骨折线方向 (line_h/edge_diag_45)')
    lines.append('     公式B: 检测骨碎片 (blob_detect/white_tophat)')
    lines.append('     公式C: 检测移位程度 (displace_ind/tb_asymmetry)')
    lines.append('     --> 分类器综合A+B+C的信息做出最终判断')
    lines.append('')
    lines.append('  4. 具体叠合示例:')
    lines.append('     如果要区分"横断骨折"和"斜行骨折":')
    lines.append('     - line_h的响应: 横断>>斜行 (水平线检测器对横断骨折响应强)')
    lines.append('     - edge_diag_45的响应: 斜行>>横断 (对角边缘检测器对斜行骨折响应强)')
    lines.append('     - 分类器学到: line_h高 + edge_diag_45低 --> 横断骨折')
    lines.append('                   line_h低 + edge_diag_45高 --> 斜行骨折')
    lines.append('')
    lines.append('  5. 统计量的作用:')
    lines.append('     同一个公式的不同统计量捕获不同信息:')
    lines.append('     - mean: 整体骨折信号强度 (骨折越明显, mean越高)')
    lines.append('     - std: 骨折区域变异度 (骨折线越不规则, std越高)')
    lines.append('     - skewness: 信号偏斜方向 (骨折线偏亮/偏暗)')
    lines.append('     - entropy: 纹理紊乱程度 (粉碎骨折entropy最高)')
    lines.append('     - 不同区域的同一统计量: 定位骨折位置')
    lines.append('')

    # ===== Part 5: Summary decision flow =====
    lines.append('=' * 90)
    lines.append('  第五部分: 综合诊断决策流')
    lines.append('=' * 90)
    lines.append('')
    lines.append('  +-------------------------------------------------------------------+')
    lines.append('  |  X光图像输入                                                       |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段1] 预处理通道                                                |')
    lines.append('  |    I_BONE(骨增强) / I_NEG(反相) / I_EDGE_PRIOR(边缘先验)           |')
    lines.append('  |    I_GRAY(灰度) / I_H(色调) / I_S(饱和度)                         |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段1] 算子组合 (RL搜索得到的最优公式)                           |')
    lines.append('  |    |                                                               |')
    lines.append('  |    +-- 骨折线检测: edge_mag / black_tophat / discont_map          |')
    lines.append('  |    +-- 方向分析: line_h / line_45 / edge_diag_45 / gabor_0        |')
    lines.append('  |    +-- 骨碎片检测: blob_detect / white_tophat                     |')
    lines.append('  |    +-- 移位检测: displace_ind / tb_asymmetry / lr_asymmetry       |')
    lines.append('  |    +-- 皮质分析: cortical_cont / bone_enhance / soft_suppress     |')
    lines.append('  |    |                                                               |')
    lines.append('  |    输出: 每个公式 -> 2D特征图[B,H,W]                              |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段2] 统计编码                                                  |')
    lines.append(f'  |    每个特征图 -> {n_stats}统计量 x {n_regions}区域 = {stats_per_formula}维向量            |')
    lines.append(f'  |    {len(formulas)}个公式 -> {len(formulas)*stats_per_formula}维原始特征空间                       |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段3] 特征选择 (ANOVA+MI)                                      |')
    if clf_info:
        lines.append(f'  |    {len(formulas)*stats_per_formula}维 -> {clf_info["n_selected"]}维 (选出最有判别力的特征)          |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段4] HistGradientBoosting分类器                                |')
    lines.append('  |    多棵决策树，每棵按特征阈值分裂                                  |')
    lines.append('  |    |                                                               |')
    lines.append('  |  [阶段5] 类别预测                                                  |')
    lines.append('  |    +-- Comminuted(粉碎):   blob_detect高 + white_tophat高 + entropy高 |')
    lines.append('  |    +-- Greenstick(青枝):   cortical_cont低(一侧) + lr_asymmetry高   |')
    lines.append('  |    +-- Healthy(健康):      cortical_cont高 + entropy低 + std低     |')
    lines.append('  |    +-- Linear(线形):       line_h/line_v高 + black_tophat高        |')
    lines.append('  |    +-- Oblique(斜行):      edge_diag_45/135高 + gabor_45高         |')
    lines.append('  |    +-- Oblique Disp(斜移): edge_diag高 + displace_ind高 + tb_asym高 |')
    lines.append('  |    +-- Segmental(节段):    discont_map多处 + entropy高              |')
    lines.append('  |    +-- Spiral(螺旋):       edge_diag_45/135高 + gabor_mag高        |')
    lines.append('  |    +-- Transverse(横断):   line_h高 + edge_y高 + gabor_0高         |')
    lines.append('  |    +-- Trans Disp(横移):   line_h高 + displace_ind高 + tb_asym高   |')
    lines.append('  +-------------------------------------------------------------------+')
    lines.append('')

    report_text = '\n'.join(lines)
    report_path = output_dir / 'causal_chain_interpretability_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f'\n  报告已保存至: {report_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Causal Chain Interpretability Report')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/fracture_v3_expanded')
    args = parser.parse_args()
    generate_report(args.output_dir)
