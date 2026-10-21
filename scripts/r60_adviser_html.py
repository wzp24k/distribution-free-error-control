# -*- coding: utf-8 -*-
"""Chunk A: header, language table (zh)."""

import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEMO_UNIT = os.path.join(ROOT, "work", "predictions",
                         "cellsplit_Melanoma_Oct2522_fold0__hgb.npz")
OUT = {l: os.path.join(ROOT, "docs", "adviser.html" if l == "zh" else "adviser_en.html")
       for l in ("zh", "en")}
SAMPLE_OUT = os.path.join(ROOT, "docs", "adviser_sample.csv")

SAMPLE_SEED = 20260928
SAMPLE_N_CAL, SAMPLE_N_DEP, SAMPLE_N_LAB = 800, 1200, 400
SAMPLE_C0, SAMPLE_SEV, SAMPLE_POW = 0.003, 0.9, 3.5

TEXT = {}
TEXT["zh"] = {
 'meta_title': '细胞注释决策向导',
 'h1': '细胞注释决策向导',
 'lead': '这个页面问你 3 个问题，然后给你一张“照做就行”的清单：哪些细胞能放心自动用、哪些必须人工看、要补多少标签、标签怎么抽。<b>不需要统计学背景。</b>',
 'quick': '最快路径：点 <b>① 用论文示例走一遍</b> → 点 <b>10% · 默认</b> → 点 <b>开始计算</b>。30 秒看完整流程。',
 'quick2': '手上已有数据：点 <b>② 上传我自己的分数文件</b>，按提示贴三列即可。没有数据：先点 <b>下载示例 CSV</b> 拿一份合成数据练手。',

 'st_s0': '开始', 'st_sUp': '数据', 'st_s1': '预算', 'st_s2': '计算', 'st_s3': '结果', 'st_s4': '行动',

 's0_step': '开始',
 's0_h': '先选一个入口',
 's0_why': '<b>为什么问这个：</b>后面的建议取决于你有没有“标签”（人工确认过的正确类型）。有标签能算得更准，没标签也能给保守建议。',
 's0_b1': '① 用论文示例走一遍',
 's0_b1s': '黑色素瘤切片、随机划分、1,843 个标定细胞 + 1,843 个待注释细胞（第一次请点这个）',
 's0_b2': '② 上传我自己的分数文件',
 's0_b2s': 'CSV 三列：set, score, error（set 填 cal 或 dep）',
 's0_b3': '③ 我什么都不懂，直接讲逻辑',
 's0_b3s': '不填数据，按论文默认设置（错误预算 10%）把流程讲一遍',
 's0_prev_h': '你会得到什么？就下面这四样，一页纸：',
 'prev_accept': '自动注释', 'prev_defer': '要人工复核', 'prev_lam': '阈值 λ', 'prev_ratio': '审计比值',
 's0_prev_note': '（这是论文示例关卡在 10% 预算下的实测值，你的数据会换成你的数字。）',

 'sUp_step': '第 1 步 · 数据',
 'sUp_h': '把你的分数贴进来',
 'sUp_why': '<b>为什么需要它：</b>要判断“哪些细胞可以自动注释”，需要知道模型对每个细胞有多不确定，以及“错误”和“不确定性”之间的关系。',
 'sUp_s1': '<b>score 怎么来？</b>用 <code>1 − 模型给出的最大类别概率</code>，越大＝模型越没把握（0＝很确定，1＝完全没把握）。',
 'sUp_s2': '<b>error 怎么填？</b>人工确认过的细胞填 <code>1</code>（模型错）或 <code>0</code>（模型对）；没确认过的<b>留空</b>。',
 'sUp_rules': '表头必须是 <code>set,score,error</code>；<code>cal</code>＝标定集（<b>每一行都必须有 error</b>），<code>dep</code>＝要预测的那一批。每边至少 30 行，cal 建议 ≥ 400 行。',
 'btn_load': '读入并继续', 'btn_template': '下载空模板', 'btn_sample': '下载示例 CSV（合成数据）',
 'btn_sample_try': '用示例数据试一遍',
 'btn_back': '← 上一步',

 's1_step': '第 2 步 · 错误预算',
 's1_h': '你能接受多少错误？',
 's1_why': '<b>为什么问这个：</b>这是唯一必须由你定的参数。它不花标签，本质是“用准确率换人工量”：预算越松，人工看得越少，写进结果的错误越多。',
 's1_how': '<b>怎么选？</b>拿不准就用 10%。下游分析对错分敏感（比如做差异表达、找 marker）就用 5%；只是粗略探索、想要最少人工就用 20%。下面这张表是论文在四个队列、570 次评估上的实测，供你对照：',
 'th_budget': '错误预算', 'th_review': '需要人工复核', 'th_risk': '保留标签里的错误率',
 'b_tight': '5% · 很严', 'b_def': '10% · 默认（推荐）', 'b_loose': '20% · 宽松',
 'b_tight_s': '人工复核 65.3%，保留标签错误率 0.021',
 'b_def_s': '人工复核 40.3%，保留标签错误率 0.061',
 'b_loose_s': '人工复核 13.6%，保留标签错误率 0.139',
 'row_tight': '5%（很严）', 'row_def': '10%（默认）', 'row_loose': '20%（宽松）',

 's2_step': '第 3 步 · 计算',
 's2_h': '点一下，剩下交给页面',
 's2_why': '<b>它在替你做什么：</b>下面三件事，一次算完，你不用懂细节。',
 's2_li1': '在标定集上找一个<span class="term" onclick="define(\'lam\')">阈值 λ</span>，使“自动注释部分”的错误率有 90% 把握不超过你选的<span class="term" onclick="define(\'alpha\')">错误预算</span>；',
 's2_li2': '把超过阈值的细胞排成<span class="term" onclick="define(\'queue\')">复核队列</span>（最没把握的排最前）；',
 's2_li3': '如果有标签，再算<span class="term" onclick="define(\'ratio\')">审计比值</span>，检查标定样本能否代表这批数据。',
 'btn_run': '开始计算', 'btn_run_s': '点完立刻出结果，不联网、不上传',

 's3_step': '第 4 步 · 结果',
 'k_accept': '自动注释（写进结果）', 'k_defer': '要人工复核', 'k_lam': '阈值 λ',
 'k_drisk': '复核队列里的错误率（本批实测）', 'k_drisk_nolab': '复核队列里的错误率（需标签）',
 'th_action': '做法', 'th_yours': '你这批数据', 'th_paper': '论文基准（同类批次）',
 'r_review': '人工复核量', 'r_c25': '队列前 1/4 能抓到的错误', 'r_c50': '队列前 1/2 能抓到的错误',
 'r_head': '队列最前 1/4 的错误率', 'r_tail': '队列最后 1/4 的错误率', 'r_ratio': '审计比值（应 ≤ 1）',
 'q_h': '复核队列长什么样',
 'cur_c25': '前 1/4 抓到错误', 'cur_c50': '前 1/2 抓到错误',
 'cur_k25': '对照：随便看 1/4', 'cur_k50': '对照：随便看 1/2',
 'res_note_lab': '“你这批数据”列是按你文件里的标签实测的。',
 'res_note_nolab': '你的文件没有部署标签，所以“你这批数据”列留空，可用论文数字作预期。',
 'res_note_part': '“你这批数据”列只用了 <b>{n}</b> 个已标注细胞（其余 {m} 个没填 error，已排除）。',
 'res_risk': '论文基准下，保留标签里的错误率约为 {e}。',
 'res_read_h': '三个数怎么解读（一句话版）',
 'res_read': '<b>{a} 自动注释</b>：这部分错误率有 90% 把握不超过 {b}。<br><b>{c} 要人看</b>：这就是人工量，可以直接排班。<br><b>{d} 阈值</b>：分数不超过它的自动通过，不用你手动设。'
            '<br>两者相加就是全批：每颗细胞要么自动通过、要么进复核队列，没有第三种去处；保证只覆盖前一部分。',
 'btn_copy': '复制结论（一段话）', 'btn_print': '打印 / 存 PDF', 'btn_json': '下载结果 JSON', 'btn_restart': '换一批数据重来',

 's4_step': '第 5 步 · 照做就行',
 's4_h': '行动清单',
 's4_why': '<b>怎么读：</b>从上往下做。打印出来贴在实验台上也行（点上面的“打印 / 存 PDF”）。',
 'chk_1': '<b>自动注释的细胞</b>：直接写进结果，它们满足你选的错误预算。',
 'chk_2': '<b>复核队列</b>：从“最没把握”往下降序看，<b>不要</b>按图片或切片顺序看。',
 'chk_3': '<b>要缩短复核</b>：从队列<b>末尾</b>砍，尾巴上的错误最稀疏。',
 'chk_4': '<b>没看的细胞不要当结论</b>：既没自动注释也没人工看的细胞，<b>没有任何保证</b>。',
 'chk_5': '<b>想提高人工效率</b>：先看队列前 1/4，能抓到约四成残余错误；看一半约六成半。',
 'chk_6': '<b>审计比值 &gt; 1</b>：标定样本不代表这批数据，必须补标签重新标定。',
 'chk_7': '<b>补标签要均匀抽</b>：从整批随机抽，<b>绝对不要</b>从复核队列里抽。',
 'chk_8': '<b>报价参考（论文 Table 6，10% 预算）</b>：12 个跨组织配对里，4 对约 500 个目标域标签就够修，3 对要 750–2,000，剩下 <b>5 对</b>不值得花标签（它们的 oracle 本身就接近 0）。',
 'red_h': '三条红线',
 'red_1': '1. 复核队列不能当标定集用 —— 论文里 180 个单元有 178 个因此失效。',
 'red_2': '2. 保证只覆盖“自动注释 + 你已看过”的细胞，没看的细胞不在保证范围内。',
 'red_3': '3. 错误预算是承诺不是目标：调大预算不会让模型变准，只会让更多错标签过关。',

 'gl_h': '看不懂就点这里',
 'gl_sum': '术语小抄（7 个词，白话版）', 'gl_w': '词', 'gl_e': '白话解释',
 'g1n': '分数 score', 'g1': '1 − 模型最大概率。0＝很确定，1＝完全没把握。',
 'g2n': '阈值 λ', 'g2': '一条线：分数不超过它＝自动注释；超过＝交人看。页面自动算出来，你不用手设。',
 'g3n': '错误预算 α', 'g3': '你允许“自动注释里”最多有多少比例错。它不花标签，是免费旋钮。',
 'g4n': '把握程度（δ=0.10）', 'g4': '“错误率不超过 α”这句话有 90% 把握成立，剩下 10% 是抽签运气。',
 'g5n': '复核队列', 'g5': '被交给人看的细胞集合，<b>数量就是人工量</b>。',
 'g6n': 'Oracle（上限参考）', 'g6': '事先知道答案时能做到的最好成绩，用来衡量还差多少。',
 'g7n': '审计比值', 'g7': '证书接受率 ÷ oracle 接受率。有效时 ≤ 1；<b>大于 1 就是“标定样本不代表这批数据”的铁证</b>。',
 'faq_sum': '常见问题（6 问）',
 'faq_1': '<b>1. 为什么显示“现在还不能自动注释”？</b><br>在标定集上找不到任何满足该预算的阈值——模型排序太弱，或标定细胞太少。三个办法：放宽预算、增加标定细胞、换更强的模型。',
 'faq_2': '<b>2. “整批都能自动注释”是什么意思？</b><br>标定样本自身的错误率已经低于你选的预算，所以把阈值放宽到全部接受仍然成立。此时人工量是 0。',
 'faq_3': '<b>3. 标定集要多少细胞？</b><br>算法最少要 25 个才有解；实践上 <b>≥400 个</b>标注细胞接受率才会脱离 0（论文 Table 6：跨组织修复通常要 500–2,000 个目标域标签）。',
 'faq_4': '<b>4. 审计比值算不出来？</b><br>它需要部署批次的标签来算 oracle。没有标签就跳过，用论文基准 0.744 / 0.766 / 3.05 作参考。',
 'faq_5': '<b>5. 不想写代码，只用这个页面行吗？</b><br>行，页面就是完整流程。要批量跑用命令行版 <code>scripts/r59_adviser.py</code>（同一套规则）。',
 'faq_6': '<b>6. 复核队列为什么不能拿来标定？</b><br>它是按分数挑出来的，不是随机样本；论文 180 个单元里 178 个因此失效。',
 'tech_sum': '数字怎么算的（给审稿人 / 技术同事）',
 'tech_1': '阈值规则与论文实现一致：在“接受数量”网格上取 <code>λ̂ = max{λ : N⁺(λ) ≤ α·D⁻(λ)}</code>，N⁺ 是“接受且错误”的细胞数对全部标定细胞数 n 的 Clopper–Pearson 单侧<b>上</b>限，D⁻ 是接受数的单侧<b>下</b>限，两侧各用 Bonferroni 份额 <code>δ/(2m)</code>（m＝网格点数），δ=0.10，最少接受 25 个标定细胞，网格 120 点。',
 'tech_2': '页面内的不完全贝塔函数用连分式＋二分实现，已与 scipy 比对（最大偏差 &lt; 1e-8），并复现论文控制器结果（示例单元 α=0.10：λ=0.1899、接受率 0.7401；α=0.05：λ=0.0371；α=0.20：λ=0.7311、接受率 1.0000）。',
 'tech_3': '预算权衡表来自 <code>experiments/r24_efficiency.csv</code>；复核队列曲线来自 <code>experiments/r57_triage.csv</code>；修复报价来自论文 Table 6；审计比值基准 0.744 / 0.766 / 3.05 来自论文 Table 4 与正文。',
 'foot_1': '配套论文：<i>Distribution-free error control for cell type annotation in spatial proteomics</i>。',
 'foot_2': '单文件、离线运行、不联网：你的 CSV 只在浏览器内存里，页面不会把它发到任何地方。',
 'foot_3': '自检：页面算法已与论文实现 <code>src/errctl/conformal.py</code> 逐步比对一致。',

 'v_ok_h': '可以用，但要按队列复核',
 'v_ok_s': '自动注释那部分的错误率有 90% 把握不超过 {a}；交给人看的细胞已按“最没把握”排好序。',
 'v_none_h': '这批数据现在还不能自动注释',
 'v_none_s': '在标定集上找不到任何满足该错误预算的阈值：模型排序太弱，或标定细胞太少。建议放宽预算、增加标定细胞，或换更强的模型。',
 'v_all_h': '整批都可以自动注释',
 'v_all_s': '标定样本自身的错误率已经低于你选的预算，把阈值放宽到全部接受仍然成立——这一批不需要人工。',
 'v_warn_h': '自动注释可用，但标定样本不代表这批数据',
 'v_warn_s': '审计比值 {r} &gt; 1，这是“标定样本与部署批次不同分布”的铁证：请补均匀抽取的标签重新标定。',
 'm_toast_copy': '已复制到剪贴板', 'm_toast_run': '先点“开始计算”',
 'm_copy_prompt': '复制下面这段（Ctrl+C）：',
 'm_empty': '还没有内容，先把 CSV 贴进来。',
 'm_small': '只解析到 cal {a} 行、dep {b} 行，各需 ≥30。请检查表头是否为 set,score,error。',
 'm_calerr': '标定集(cal)每行都必须有 error，且只能是 0 或 1。',
 'm_toofew': '标定细胞太少（算法至少需要 {n} 个）',
 'lab_demo': '论文示例（黑色素瘤，随机划分）',
 'lab_logic': '逻辑演示（未填数据，按论文基准）',
 'lab_mine': '我的数据（cal {a} 行，dep {b} 行）',
 'lab_mine_lab': '，含标签', 'lab_mine_part': '，部分含标签（{n} 个）', 'lab_mine_nolab': '，无标签',
 'c_title': '细胞注释决策结论', 'c_data': '数据：', 'c_budget': '错误预算：',
 'c_budget_note': '（自动注释部分的错误率有 90% 把握不超过它）',
 'c_lam': '阈值 λ = ', 'c_accept': '自动注释：', 'c_defer': '　需人工复核：',
 'c_drisk': '复核队列实测错误率：', 'c_ratio': '审计比值：',
 'c_ratio_bad': '（>1：标定样本不代表这批数据，需补标签重标定）', 'c_ratio_ok': '（≤1：标定样本可用）',
 'c_action': '行动：自动注释直接采用；复核队列从最没把握往下降序看，要缩短就从队列末尾砍；没看的细胞不写进结论。',
 'c_none': '（无解）',
 'noplan': '先点“开始计算”',
}

TEXT['en'] = {
 'meta_title': 'Cell-annotation decision guide',
 'h1': 'Cell-annotation decision guide',
 'lead': 'This page asks you three questions and then gives you a checklist you can follow: which cells can be labelled automatically, which ones a human must check, how many labels to buy, and how to draw them. <b>No statistics background is needed.</b>',
 'quick': 'Fastest path: press <b>1 · Walk through the paper’s example</b> → press <b>10% · default</b> → press <b>Compute</b>. The whole procedure takes about thirty seconds.',
 'quick2': 'Already have data: press <b>2 · Upload my own score file</b> and paste the three columns. No data at all: press <b>Download an example CSV</b> for a synthetic batch to practise on.',

 'st_s0': 'Start', 'st_sUp': 'Data', 'st_s1': 'Budget', 'st_s2': 'Compute', 'st_s3': 'Result', 'st_s4': 'Act',

 's0_step': 'Start',
 's0_h': 'Pick an entry point',
 's0_why': '<b>Why this is asked:</b> the advice depends on whether you have <i>labels</i> (cell types a human has confirmed). With labels the numbers are sharper; without them the page still gives a conservative answer.',
 's0_b1': '1 · Walk through the paper’s example',
 's0_b1s': 'melanoma section, random split, 1,843 calibration cells and 1,843 cells to annotate (start here the first time)',
 's0_b2': '2 · Upload my own score file',
 's0_b2s': 'CSV with three columns: set, score, error (set is cal or dep)',
 's0_b3': '3 · I know nothing — just explain the logic',
 's0_b3s': 'no data; walks the procedure at the paper’s default (a 10% error budget)',
 's0_prev_h': 'What you get. Four numbers, one page:',
 'prev_accept': 'labelled automatically', 'prev_defer': 'to a human', 'prev_lam': 'threshold λ', 'prev_ratio': 'audit ratio',
 's0_prev_note': '(measured on the paper’s example unit at a 10% budget; your data will give your own numbers.)',

 'sUp_step': 'Step 1 · Data',
 'sUp_h': 'Paste your scores',
 'sUp_why': '<b>Why it is needed:</b> deciding which cells can be labelled automatically requires knowing how unsure the model is about each cell, and how error relates to that uncertainty.',
 'sUp_s1': '<b>Where does the score come from?</b> Use <code>1 − the largest class probability the model returned</code>. Larger means less sure (0 = certain, 1 = no idea).',
 'sUp_s2': '<b>What goes in error?</b> For a cell a human has confirmed, write <code>1</code> (the model was wrong) or <code>0</code> (the model was right). For a cell nobody has checked, <b>leave it blank</b>.',
 'sUp_rules': 'The header must be <code>set,score,error</code>; <code>cal</code> is the calibration set (<b>every row must carry an error</b>) and <code>dep</code> is the batch to annotate. At least 30 rows each side; for cal, 400 or more is advisable.',
 'btn_load': 'Read it and continue', 'btn_template': 'Download an empty template',
 'btn_sample': 'Download an example CSV (synthetic)',
 'btn_sample_try': 'Try it on the example data',
 'btn_back': '← Back',

 's1_step': 'Step 2 · Error budget',
 's1_h': 'How much error can you accept?',
 's1_why': '<b>Why this is asked:</b> it is the only parameter you have to choose. It costs no labels: it trades accuracy for human effort. A looser budget means less to read and more error written into the result.',
 's1_how': '<b>How to choose.</b> If unsure, take 10%. Take 5% when downstream analysis is sensitive to mislabelling (differential expression, marker discovery); take 20% for rough exploration where you want the least possible reading. The table is what the paper measured over 570 evaluations in four cohorts:',
 'th_budget': 'Error budget', 'th_review': 'Needs human review', 'th_risk': 'Error among retained labels',
 'b_tight': '5% · very strict', 'b_def': '10% · default (recommended)', 'b_loose': '20% · permissive',
 'b_tight_s': '65.3% reviewed, 0.021 error among retained labels',
 'b_def_s': '40.3% reviewed, 0.061 error among retained labels',
 'b_loose_s': '13.6% reviewed, 0.139 error among retained labels',
 'row_tight': '5% (very strict)', 'row_def': '10% (default)', 'row_loose': '20% (permissive)',

 's2_step': 'Step 3 · Compute',
 's2_h': 'Press the button; the page does the rest',
 's2_why': '<b>What it does for you:</b> the three things below, in one pass. You do not need the details.',
 's2_li1': 'find a <span class="term" onclick="define(\'lam\')">threshold λ</span> on the calibration set so that the error among the automatically labelled cells is ≤ your <span class="term" onclick="define(\'alpha\')">error budget</span> with 90% confidence;',
 's2_li2': 'sort the cells above that threshold into a <span class="term" onclick="define(\'queue\')">review queue</span>, least confident first;',
 's2_li3': 'if labels exist, compute the <span class="term" onclick="define(\'ratio\')">audit ratio</span> — whether the calibration sample can stand in for this batch.',
 'btn_run': 'Compute', 'btn_run_s': 'instant, offline, nothing uploaded',

 's3_step': 'Step 4 · Result',
 'k_accept': 'labelled automatically (written to the result)', 'k_defer': 'to a human',
 'k_lam': 'threshold λ',
 'k_drisk': 'error in the review queue (measured here)', 'k_drisk_nolab': 'error in the review queue (needs labels)',
 'th_action': 'Quantity', 'th_yours': 'Your batch', 'th_paper': 'Paper benchmark (same family)',
 'r_review': 'Human review', 'r_c25': 'Errors caught in the first quarter of the queue',
 'r_c50': 'Errors caught in the first half', 'r_head': 'Error rate, first quarter of the queue',
 'r_tail': 'Error rate, last quarter', 'r_ratio': 'Audit ratio (should be ≤ 1)',
 'q_h': 'What the review queue looks like',
 'cur_c25': 'first quarter', 'cur_c50': 'first half',
 'cur_k25': 'control: 1/4 at random', 'cur_k50': 'control: 1/2 at random',
 'res_note_lab': 'The “Your batch” column is measured on the labels in your file.',
 'res_note_nolab': 'Your file carries no deployment labels, so the “Your batch” column is empty; use the paper’s numbers as the expectation.',
 'res_note_part': 'The “Your batch” column uses only the <b>{n}</b> labelled cells; the other {m} have no error filled in and were excluded.',
 'res_risk': 'On the paper’s benchmark the error among retained labels is about {e}.',
 'res_read_h': 'The three numbers, one line each',
 'res_read': '<b>{a} labelled automatically</b>: the error among them is ≤ {b} with 90% confidence.<br><b>{c} to a human</b>: this is your reading list, and it can be scheduled as it stands.<br><b>threshold {d}</b>: cells scoring at or below it pass automatically; you never set it by hand.'
            '<br>The two add up to the whole batch: every cell either passes automatically or lands in'
            ' your review queue, and there is no third disposition. The guarantee covers only the'
            ' first part.',
 'btn_copy': 'Copy the conclusion', 'btn_print': 'Print / save as PDF', 'btn_json': 'Download JSON', 'btn_restart': 'Start again with other data',

 's4_step': 'Step 5 · Do this',
 's4_h': 'Action list',
 's4_why': '<b>How to use it.</b> Work down the list. Print it and tape it to the bench if that helps (use “Print / save as PDF” above).',
 'chk_1': '<b>Automatically labelled cells</b>: write them into the result. They satisfy the error budget you chose.',
 'chk_2': '<b>The review queue</b>: read it in descending uncertainty. <b>Do not</b> read in image or section order.',
 'chk_3': '<b>To shorten the review</b>: cut from the <b>end</b> of the queue, where errors are scarcest.',
 'chk_4': '<b>A cell nobody looked at carries no conclusion</b>: neither automatically labelled nor reviewed means no guarantee at all.',
 'chk_5': '<b>To work faster</b>: read the first quarter of the queue and you capture about two fifths of the residual errors; half the queue captures about two thirds.',
 'chk_6': '<b>Audit ratio &gt; 1</b>: the calibration sample does not represent this batch; labels must be bought and the certificate refitted.',
 'chk_7': '<b>Draw new labels uniformly</b>: at random from the whole batch, <b>never</b> from the review queue.',
 'chk_8': '<b>Price list (Table 6, 10% budget)</b>: of the 12 cross-tissue pairs, 4 reach validity with about 500 target labels, 3 need 750–2,000, and the remaining <b>5</b> are not worth the labels — their oracle is already near zero.',
 'red_h': 'Three hard rules',
 'red_1': '1. The review queue is not a calibration set — in the paper 178 of 180 units fail through exactly this substitution.',
 'red_2': '2. The guarantee covers the automatically labelled cells and the cells you have reviewed. It covers nothing else.',
 'red_3': '3. The budget is a commitment, not a target: raising it does not make the model better, it lets more wrong labels through.',

 'gl_h': 'Open this if a word is unfamiliar',
 'gl_sum': 'Glossary (7 terms, in plain words)', 'gl_w': 'Term', 'gl_e': 'In plain words',
 'g1n': 'score', 'g1': '1 − the model’s largest probability. 0 = certain, 1 = no idea.',
 'g2n': 'threshold λ', 'g2': 'A line: a cell scoring at or below it is labelled automatically, above it goes to a human. The page computes it; you never set it by hand.',
 'g3n': 'error budget α', 'g3': 'The largest fraction of wrong labels you will accept among the automatically labelled cells. It costs no labels.',
 'g4n': 'confidence (δ = 0.10)', 'g4': 'The sentence “the error is at most α” holds with 90% confidence; the other 10% is the luck of the draw.',
 'g5n': 'review queue', 'g5': 'The cells handed to a human. <b>Its size is your human effort.</b>',
 'g6n': 'oracle (upper reference)', 'g6': 'What the best possible threshold achieves when the answers are known; it measures the distance still to be closed.',
 'g7n': 'audit ratio', 'g7': 'Certificate acceptance ÷ oracle acceptance. At most 1 when the certificate is valid; <b>above 1 is proof that the calibration sample does not represent this batch</b>.',
 'faq_sum': 'Frequently asked (6)',
 'faq_1': '<b>1. Why does it say no cell can be labelled automatically?</b><br>On the calibration set there is no threshold that meets the budget — the model ranks too weakly, or there are too few calibration cells. Three remedies: loosen the budget, buy more calibration cells, wrap a stronger model.',
 'faq_2': '<b>2. What does “the whole batch can be labelled automatically” mean?</b><br>The calibration sample’s own error is already below the budget you chose, so relaxing the threshold all the way still holds. Human effort is then zero.',
 'faq_3': '<b>3. How many calibration cells do I need?</b><br>The algorithm needs at least 25 before a solution exists; in practice <b>400 or more</b> labelled cells before acceptance leaves zero (Table 6: a cross-tissue repair usually costs 500–2,000 target labels).',
 'faq_4': '<b>4. The audit ratio will not compute.</b><br>It needs deployment labels to define the oracle. Without them the page skips it; use the paper’s 0.744 / 0.766 / 3.05 as the benchmark.',
 'faq_5': '<b>5. Can I use this page and never write code?</b><br>Yes — the page is the whole procedure. To run it in batch, use the command line, <code>scripts/r59_adviser.py</code>, which implements the same rule.',
 'faq_6': '<b>6. Why can’t the review queue be used for calibration?</b><br>It is selected by score, so it is not a random sample; in the paper 178 of 180 units fail for this reason.',
 'tech_sum': 'How the numbers are computed (for referees and technical colleagues)',
 'tech_1': 'The threshold rule is the paper’s: over a grid of acceptance counts take <code>λ̂ = max{λ : N⁺(λ) ≤ α·D⁻(λ)}</code>, where N⁺ is a one-sided Clopper–Pearson <b>upper</b> limit on the accepted-and-wrong count out of all n calibration cells and D⁻ is a one-sided <b>lower</b> limit on the accepted count, each charged a Bonferroni share <code>δ/(2m)</code> over the m grid points, with δ = 0.10, a floor of 25 accepted calibration cells and a 120-point grid.',
 'tech_2': 'The incomplete beta function is evaluated in the page by continued fraction and bisection; it agrees with scipy to better than 1e-8 and reproduces the paper’s controller on the embedded unit (α = 0.10: λ = 0.1899, acceptance 0.7401; α = 0.05: λ = 0.0371; α = 0.20: λ = 0.7311, acceptance 1.0000).',
 'tech_3': 'The budget table is from <code>experiments/r24_efficiency.csv</code>, the queue curve from <code>experiments/r57_triage.csv</code>, the repair price list from Table 6, and the audit-ratio benchmark 0.744 / 0.766 / 3.05 from Table 4 and the main text.',
 'foot_1': 'Companion paper: <i>Distribution-free error control for cell type annotation in spatial proteomics</i>.',
 'foot_2': 'One file, offline, no network: your CSV never leaves the browser’s memory.',
 'foot_3': 'Self-check: the page’s arithmetic is compared against the paper’s implementation, <code>src/errctl/conformal.py</code>, step by step.',

 'v_ok_h': 'Usable, but work through the queue',
 'v_ok_s': 'The error among the automatically labelled cells is at most {a} with 90% confidence; the cells handed to a human are already ordered least-confident-first.',
 'v_none_h': 'This batch cannot be labelled automatically yet',
 'v_none_s': 'No threshold on the calibration set meets that budget: either the model ranks too weakly or there are too few calibration cells. Loosen the budget, add calibration cells, or wrap a stronger model.',
 'v_all_h': 'The whole batch can be labelled automatically',
 'v_all_s': 'The calibration sample’s own error is already below the budget, so a threshold that accepts everything still holds. This batch needs no human.',
 'v_warn_h': 'Usable, but the calibration sample does not represent this batch',
 'v_warn_s': 'The audit ratio is {r}, above 1, which is proof that the calibration sample and the deployment batch are differently distributed: draw labels uniformly and refit.',
 'm_toast_copy': 'Copied to the clipboard', 'm_toast_run': 'Press “Compute” first',
 'm_copy_prompt': 'Copy this (Ctrl+C):',
 'm_empty': 'Nothing here yet — paste the CSV first.',
 'm_small': 'Parsed {a} cal rows and {b} dep rows; at least 30 of each are needed. Check that the header is set,score,error.',
 'm_calerr': 'Every calibration (cal) row must carry an error, and it must be 0 or 1.',
 'm_toofew': 'too few calibration cells (the rule needs at least {n})',
 'lab_demo': 'paper example (melanoma, random split)',
 'lab_logic': 'logic walk-through (no data; paper benchmark)',
 'lab_mine': 'my data ({a} cal rows, {b} dep rows)',
 'lab_mine_lab': ', with labels', 'lab_mine_part': ', partly labelled ({n} cells)',
 'lab_mine_nolab': ', no labels',
 'c_title': 'Cell-annotation decision',
 'c_data': 'Data: ', 'c_budget': 'Error budget: ',
 'c_budget_note': ' (the error among automatically labelled cells is at most this with 90% confidence)',
 'c_lam': 'threshold λ = ', 'c_accept': 'labelled automatically: ', 'c_defer': '   to a human: ',
 'c_drisk': 'error in the review queue (measured): ', 'c_ratio': 'audit ratio: ',
 'c_ratio_bad': ' (> 1: the calibration sample does not represent this batch; buy labels and refit)',
 'c_ratio_ok': ' (≤ 1: the calibration sample stands)',
 'c_action': 'Action: adopt the automatically labelled cells; read the queue in descending uncertainty and, to shorten it, cut from the end; write no conclusion about cells nobody has seen.',
 'c_none': '(no solution)',
 'noplan': 'Press “Compute” first',
}

TEXT["zh"].update({
 'v_logic_h': '这一页给的是逻辑演示',
 'v_logic_s': '先在标定集上定阈值，再按不确定度排序人工复核，最后用标签做审计。想要你自己的数字，回第 1 步点“论文示例”或上传 CSV。',
 'lab_sample_suffix': '（合成示例数据）',
})
TEXT["en"].update({
 'v_logic_h': 'This page is showing the logic, not your numbers',
 'v_logic_s': 'Fit a threshold on the calibration set, read the queue in descending uncertainty, audit with labels. For your own numbers go back to step 1: use the paper example, or upload a CSV.',
 'lab_sample_suffix': ' (synthetic example data)',
})

SKELETON = r"""<!DOCTYPE html>
<!-- Generated by scripts/r60_adviser_html.py -- edit that script, not this file. -->
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#1d2333;--sub:#5b6579;--line:#e4e7f0;--blue:#2f6df6;--bl:#eaf0ff;--gr:#148f5b;--grl:#e7f7ef;--am:#b7791f;--aml:#fdf6e6;--rd:#c0392b;--rdl:#fdeeec;--sh:0 6px 22px rgba(23,32,64,.08)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:26px 18px 80px}
h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}h3{font-size:16.5px;margin:22px 0 8px}
.lead,.muted{color:var(--sub)}.lead{margin:0 0 16px;font-size:16.5px}.muted{font-size:13.5px}
.quick{background:linear-gradient(90deg,#eaf0ff,#f6f9ff);border:1px solid #cfe0ff;border-radius:14px;padding:14px 18px;margin:0 0 18px;font-size:15px}
.quick b{color:var(--blue)}
.bar{display:flex;gap:6px;margin:0 0 20px;flex-wrap:wrap}
.bar .dot{flex:1;min-width:78px;text-align:center;font-size:12.5px;color:var(--sub);background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 4px}
.bar .dot b{display:block;font-size:15px;color:#b9c0d4;line-height:1.3}
.bar .dot.done{background:var(--bl);border-color:#cfe0ff;color:#2a3a63}
.bar .dot.done b{color:var(--blue)}
.bar .dot.on{background:var(--blue);border-color:var(--blue);color:#fff}
.bar .dot.on b{color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:var(--sh);margin-bottom:18px}
.step{font-size:13px;letter-spacing:.1em;color:var(--blue);font-weight:700;margin-bottom:8px}
.why{background:var(--bl);border-left:3px solid var(--blue);border-radius:8px;padding:10px 14px;color:#2a3a63;font-size:14.5px;margin:0 0 16px}
.why b{color:#1e2a4a}
button{font:inherit;cursor:pointer;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:14px 16px;min-height:52px;text-align:left;width:100%;margin:8px 0;transition:.15s}
button:hover{border-color:var(--blue);box-shadow:0 3px 12px rgba(47,109,246,.14)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:600}
button.ghost{text-align:center;min-height:44px;color:var(--sub);font-size:14.5px}
button .small{display:block;color:var(--sub);font-size:13px;margin-top:3px;font-weight:400}
button.primary .small{color:#dfe8ff}
.row{display:flex;gap:10px;flex-wrap:wrap}.row button{flex:1;min-width:190px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:14px 0}
.kpi{background:#fbfcff;border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi b{display:block;font-size:24px;line-height:1.25}.kpi span{color:var(--sub);font-size:13px}
.verdict{border-radius:14px;padding:16px 18px;font-weight:600;margin:0 0 16px}
.verdict span{display:block;font-weight:400;color:var(--ink);margin-top:6px;font-size:14.5px}
.v-green{background:var(--grl);color:var(--gr);border:1px solid #b9e6cf}
.v-amber{background:var(--aml);color:var(--am);border:1px solid #f0dfae}
.v-red{background:var(--rdl);color:var(--rd);border:1px solid #f3c9c3}
table{width:100%;border-collapse:collapse;font-size:14.5px;margin:10px 0}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}th{color:var(--sub);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.curve div{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:14px}
.curve .lab{width:200px;color:var(--sub);flex:none}
.curve .track{flex:1;background:#eef1f8;border-radius:7px;height:22px;position:relative;overflow:hidden}
.curve .fill{position:absolute;left:0;top:0;bottom:0;background:var(--blue);border-radius:7px}.curve .fill.alt{background:#9db8f7}
.curve .pct{width:62px;text-align:right;font-variant-numeric:tabular-nums;flex:none}
ul.check{list-style:none;padding:0;margin:6px 0}
ul.check li{background:#fbfcff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:8px 0;display:flex;gap:11px;align-items:flex-start}
ul.check input{margin-top:4px;width:19px;height:19px;flex:none}
.redline{background:var(--rdl);border:1px solid #f3c9c3;border-radius:12px;padding:14px 18px;margin:14px 0}.redline b{color:var(--rd)}
details{background:#fbfcff;border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:10px 0}
details summary{cursor:pointer;font-weight:600}
details summary::marker{color:var(--blue)}
code{background:#f1f3f9;border-radius:5px;padding:1px 6px;font-size:13.5px;font-family:ui-monospace,Consolas,monospace}
textarea{width:100%;min-height:130px;font:13px/1.5 ui-monospace,Consolas,monospace;border:1px solid var(--line);border-radius:10px;padding:10px}
.term{cursor:pointer;color:var(--blue);border-bottom:1px dashed var(--blue);white-space:nowrap}
.mini{background:#fbfcff;border:1px dashed #cfe0ff;border-radius:12px;padding:14px 16px;margin:12px 0}
.mini .k{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:9px;padding:6px 10px;margin:4px 6px 4px 0;font-size:14px}
.mini .k b{color:var(--blue)}
.hide{display:none}
footer{color:var(--sub);font-size:13px;margin-top:26px;line-height:1.8}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);background:#1d2333;color:#fff;padding:10px 18px;border-radius:10px;opacity:0;pointer-events:none;transition:.25s;font-size:14.5px;z-index:9}
#toast.on{opacity:1}
@keyframes fl{0%,100%{background:transparent}30%{background:#fff3c4}}
tr.flash{animation:fl 1.7s ease}
@media print{.noprint{display:none}body{background:#fff}.card{box-shadow:none;break-inside:avoid}.bar{display:none}}
</style>
</head>
<body>
<div class="wrap">
<h1 data-t="h1"></h1>
<p class="lead" data-th="lead"></p>
<div class="quick noprint"><span data-th="quick"></span><br><span data-th="quick2"></span></div>

<div class="bar" id="bar"></div>

<div class="card" id="s0">
  <div class="step" data-t="s0_step"></div>
  <h2 data-t="s0_h"></h2>
  <p class="why" data-th="s0_why"></p>
  <button class="primary" onclick="startDemo()"><span data-t="s0_b1"></span><span class="small" data-t="s0_b1s"></span></button>
  <button onclick="show('sUp')"><span data-t="s0_b2"></span><span class="small" data-t="s0_b2s"></span></button>
  <button onclick="noData()"><span data-t="s0_b3"></span><span class="small" data-t="s0_b3s"></span></button>
  <div class="mini">
    <b data-t="s0_prev_h"></b>
    <div style="margin-top:8px">
      <span class="k"><span data-t="prev_accept"></span> <b>74.0%</b></span>
      <span class="k"><span data-t="prev_defer"></span> <b>26.0%</b></span>
      <span class="k"><span data-t="prev_lam"></span> <b>0.1899</b></span>
      <span class="k"><span data-t="prev_ratio"></span> <b>0.906</b></span>
    </div>
    <div class="muted" style="margin-top:6px" data-t="s0_prev_note"></div>
  </div>
</div>

<div class="card hide" id="sUp">
  <div class="step" data-t="sUp_step"></div>
  <h2 data-t="sUp_h"></h2>
  <p class="why" data-th="sUp_why"></p>
  <p data-th="sUp_s1"></p>
  <p data-th="sUp_s2"></p>
  <p class="muted" data-th="sUp_rules"></p>
  <textarea id="csv" spellcheck="false" placeholder="set,score,error
cal,0.02,0
cal,0.31,1
dep,0.05,
dep,0.44,"></textarea>
  <div class="row"><button class="primary" onclick="loadCSV()" data-t="btn_load"></button><button onclick="template()" data-t="btn_template"></button></div>
  <div class="row"><button onclick="downloadSample()" data-t="btn_sample"></button><button onclick="loadSample()" data-t="btn_sample_try"></button></div>
  <p id="csvmsg" class="muted"></p>
  <div class="row noprint"><button class="ghost" onclick="back()" data-t="btn_back"></button></div>
</div>

<div class="card hide" id="s1">
  <div class="step" data-t="s1_step"></div>
  <h2 data-t="s1_h"></h2>
  <p class="why" data-th="s1_why"></p>
  <div id="budgetBtns"></div>
  <p class="muted" data-th="s1_how"></p>
  <table><thead><tr><th data-t="th_budget"></th><th class="n" data-t="th_review"></th><th class="n" data-t="th_risk"></th></tr></thead>
  <tbody><tr><td data-t="row_tight"></td><td class="n">65.3%</td><td class="n">0.021</td></tr>
  <tr><td data-t="row_def"></td><td class="n">40.3%</td><td class="n">0.061</td></tr>
  <tr><td data-t="row_loose"></td><td class="n">13.6%</td><td class="n">0.139</td></tr></tbody></table>
  <div class="row noprint"><button class="ghost" onclick="back()" data-t="btn_back"></button></div>
</div>

<div class="card hide" id="s2">
  <div class="step" data-t="s2_step"></div>
  <h2 data-t="s2_h"></h2>
  <p class="why" data-th="s2_why"></p>
  <ol><li data-th="s2_li1"></li><li data-th="s2_li2"></li><li data-th="s2_li3"></li></ol>
  <div class="row"><button class="primary" onclick="run()"><span data-t="btn_run"></span><span class="small" data-t="btn_run_s"></span></button></div>
  <div class="row noprint"><button class="ghost" onclick="back()" data-t="btn_back"></button></div>
</div>

<div class="card hide" id="s3">
  <div class="step" data-t="s3_step"></div>
  <div id="result"></div>
  <div class="row noprint"><button class="primary" onclick="conclusion()" data-t="btn_copy"></button></div>
  <div class="row noprint"><button onclick="window.print()" data-t="btn_print"></button>
  <button onclick="downloadJson()" data-t="btn_json"></button><button onclick="restart()" data-t="btn_restart"></button></div>
  <div class="row noprint"><button class="ghost" onclick="back()" data-t="btn_back"></button></div>
</div>

<div class="card hide" id="s4">
  <div class="step" data-t="s4_step"></div>
  <h2 data-t="s4_h"></h2>
  <p class="why" data-th="s4_why"></p>
  <ul class="check">
    <li><input type="checkbox"><div data-th="chk_1"></div></li>
    <li><input type="checkbox"><div data-th="chk_2"></div></li>
    <li><input type="checkbox"><div data-th="chk_3"></div></li>
    <li><input type="checkbox"><div data-th="chk_4"></div></li>
    <li><input type="checkbox"><div data-th="chk_5"></div></li>
    <li><input type="checkbox"><div data-th="chk_6"></div></li>
    <li><input type="checkbox"><div data-th="chk_7"></div></li>
    <li><input type="checkbox"><div data-th="chk_8"></div></li>
  </ul>
  <div class="redline"><b data-t="red_h"></b>
    <div data-th="red_1"></div><div data-th="red_2"></div><div data-th="red_3"></div></div>
  <div class="row noprint"><button class="primary" onclick="conclusion()" data-t="btn_copy"></button></div>
  <div class="row noprint"><button class="ghost" onclick="show('s3')" data-t="btn_back"></button></div>
</div>

<div class="card">
  <h2 data-t="gl_h"></h2>
  <details id="gloss"><summary data-t="gl_sum"></summary>
  <table><tr><th data-t="gl_w"></th><th data-t="gl_e"></th></tr>
  <tr id="g-score"><td data-t="g1n"></td><td data-th="g1"></td></tr>
  <tr id="g-lam"><td data-t="g2n"></td><td data-th="g2"></td></tr>
  <tr id="g-alpha"><td data-t="g3n"></td><td data-th="g3"></td></tr>
  <tr id="g-delta"><td data-t="g4n"></td><td data-th="g4"></td></tr>
  <tr id="g-queue"><td data-t="g5n"></td><td data-th="g5"></td></tr>
  <tr id="g-oracle"><td data-t="g6n"></td><td data-th="g6"></td></tr>
  <tr id="g-ratio"><td data-t="g7n"></td><td data-th="g7"></td></tr></table>
  </details>
  <details><summary data-t="faq_sum"></summary>
  <p data-th="faq_1"></p><p data-th="faq_2"></p><p data-th="faq_3"></p>
  <p data-th="faq_4"></p><p data-th="faq_5"></p><p data-th="faq_6"></p>
  </details>
  <details><summary data-t="tech_sum"></summary>
  <p data-th="tech_1"></p><p data-th="tech_2"></p><p data-th="tech_3"></p>
  </details>
</div>

<footer><span data-th="foot_1"></span><br><span data-th="foot_2"></span><br><span data-th="foot_3"></span></footer>
</div>
<div id="toast"></div>
<script>
const T=__TEXT_JSON__;
const SAMPLE=__SAMPLE_JSON__;
const PUB={budget:{0.05:[0.653,0.021],0.10:[0.403,0.061],0.20:[0.136,0.139]},
           queue:{0.05:[0.477,0.749,0.498,0.118],0.10:[0.386,0.655,0.568,0.265],0.20:[0.309,0.568,0.794,0.594]}};
const DEMO=__DEMO__;
const STEPS=[["s0",T.st_s0],["sUp",T.st_sUp],["s1",T.st_s1],["s2",T.st_s2],["s3",T.st_s3],["s4",T.st_s4]];
const PREV={sUp:"s0",s1:"s0",s2:"s1",s3:"s2",s4:"s3"};
let CUR="s0";
let S={data:null,alpha:0.10,out:null,label:"",fromSample:false};
function fmt(t,o){return String(t).replace(/\{(\w+)\}/g,function(m,k){return (k in o)?o[k]:m;});}
document.title=T.meta_title;
document.querySelectorAll("[data-t]").forEach(function(e){e.textContent=T[e.dataset.t]||"";});
document.querySelectorAll("[data-th]").forEach(function(e){e.innerHTML=T[e.dataset.th]||"";});
function betacf(a,b,x){
  const MAXIT=200,EPS=3e-16,FPMIN=1e-300,qab=a+b,qap=a+1,qam=a-1;
  let c=1,d=1-qab*x/qap; if(Math.abs(d)<FPMIN)d=FPMIN; d=1/d; let h=d;
  for(let m=1;m<=MAXIT;m++){
    const m2=2*m;
    let aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; if(Math.abs(d)<FPMIN)d=FPMIN;
    c=1+aa/c; if(Math.abs(c)<FPMIN)c=FPMIN; d=1/d; h*=d*c;
    aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; if(Math.abs(d)<FPMIN)d=FPMIN;
    c=1+aa/c; if(Math.abs(c)<FPMIN)c=FPMIN; d=1/d; const del=d*c; h*=del;
    if(Math.abs(del-1)<EPS)break;
  }
  return h;
}
function lgamma(x){
  const g=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];
  let y=x,t=x+5.5; t-=(x+0.5)*Math.log(t);
  let ser=1.000000000190015;
  for(let j=0;j<6;j++) ser+=g[j]/++y;
  return -t+Math.log(2.5066282746310005*ser/x);
}
function betai(a,b,x){
  if(x<=0)return 0; if(x>=1)return 1;
  const bt=Math.exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*Math.log(x)+b*Math.log(1-x));
  return x<(a+1)/(a+b+2) ? bt*betacf(a,b,x)/a : 1-bt*betacf(b,a,1-x)/b;
}
function betaQuantile(p,a,b){let lo=0,hi=1;for(let i=0;i<80;i++){const m=(lo+hi)/2; if(betai(a,b,m)<p) lo=m; else hi=m;}return (lo+hi)/2;}
function cpUpper(x,n,d){return x<=0?1-Math.pow(d,1/n):betaQuantile(1-d,x+1,Math.max(n-x,1));}
function cpLower(x,n,d){return betaQuantile(d,x,n-x+1);}
function fitLambda(s,e,alpha,delta,minAccept,gridSize){
  const idx=s.map(function(v,i){return i;}).sort(function(i,j){return s[i]-s[j]||i-j;});
  const ss=idx.map(function(i){return s[i];}),ee=idx.map(function(i){return e[i];}),n=ss.length,lo=Math.max(minAccept,2);
  if(lo>n) return {lam:-Infinity,reason:fmt(T.m_toofew,{n:lo})};
  const steps=Math.min(gridSize,n-lo+1);
  const raw=[];
  for(let i=0;i<steps;i++) raw.push(steps===1?lo:Math.round(lo+(n-lo)*i/(steps-1)));
  const cand=Array.from(new Set(raw)).sort(function(a,b){return a-b;});
  const m=cand.length,dm=delta/(2*m);
  let cum=0; const running=[];
  for(let i=0;i<n;i++){cum+=ee[i];running.push(cum);}
  let lam=-Infinity,chosen=null;
  for(let j=0;j<cand.length;j++){
    const k=cand[j];
    const up=cpUpper(Math.round(running[k-1]),n,dm),low=cpLower(k,n,dm),ratio=low>0?up/low:Infinity;
    if(ratio<=alpha){lam=ss[k-1];chosen=k;}
  }
  return {lam:lam,chosen:chosen,delta_m:dm,grid:m,n:n};
}
function oracleLambda(s,e,alpha,minAccept){
  const idx=s.map(function(v,i){return i;}).sort(function(i,j){return s[i]-s[j]||i-j;});
  const ss=idx.map(function(i){return s[i];}),ee=idx.map(function(i){return e[i];});
  let cum=0,best=-Infinity;
  for(let k=1;k<=ss.length;k++){cum+=ee[k-1]; if(k>=minAccept&&cum/k<=alpha) best=ss[k-1];}
  return best;
}
function paint(id){
  STEPS.forEach(function(p){document.getElementById(p[0]).classList.add("hide");});
  document.getElementById(id).classList.remove("hide");
  const i=STEPS.map(function(p){return p[0];}).indexOf(id);
  document.getElementById("bar").innerHTML=STEPS.map(function(p,j){
    return '<span class="dot'+(j<i?' done':'')+(j===i?' on':'')+'"><b>'+(j+1)+'</b>'+p[1]+'</span>';}).join("");
  window.scrollTo({top:0,behavior:"smooth"});
}
function show(id){ if(CUR&&id!==CUR){PREV[id]=CUR;} CUR=id; paint(id); }
function back(){ let t=PREV[CUR]||"s0"; if(t===CUR)t="s0"; CUR=t; paint(t); }
function toast(msg){const el=document.getElementById("toast");el.textContent=msg;el.classList.add("on");setTimeout(function(){el.classList.remove("on");},2200);}
function define(id){
  const g=document.getElementById("gloss"); if(g) g.open=true;
  const row=document.getElementById("g-"+id); if(!row) return;
  row.scrollIntoView({block:"center",behavior:"smooth"});
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
}
function budgetCards(){
  const b=[[0.05,T.b_tight,T.b_tight_s],[0.10,T.b_def,T.b_def_s],[0.20,T.b_loose,T.b_loose_s]];
  document.getElementById("budgetBtns").innerHTML=b.map(function(x){
    return '<button class="'+(x[0]===0.10?'primary':'')+'" onclick="pick('+x[0]+')">'+x[1]+
           '<span class="small">'+x[2]+'</span></button>';}).join("");
}
function pick(a){S.alpha=a;show("s2");}
function startDemo(){
  S.data={calScores:DEMO.calScores,calErrors:DEMO.calErrors,depScores:DEMO.depScores,depErrors:DEMO.depErrors};
  S.label=T.lab_demo;S.fromSample=false;budgetCards();show("s1");
}
function noData(){S.data=null;S.label=T.lab_logic;budgetCards();show("s1");}
function template(){
  const blob=new Blob(["set,score,error\ncal,0.02,0\ncal,0.31,1\ndep,0.05,\ndep,0.44,\n"],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="scores_template.csv";a.click();
}
function downloadSample(){
  const blob=new Blob([SAMPLE],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="adviser_sample.csv";a.click();
}
function loadSample(){
  document.getElementById("csv").value=SAMPLE;
  S.fromSample=true;
  loadCSV();
}
function loadCSV(){
  const raw=document.getElementById("csv").value.trim(),msg=document.getElementById("csvmsg");
  if(!raw){msg.textContent=T.m_empty;return;}
  const lines=raw.split(/\r?\n/).filter(function(l){return l.trim();});
  const start=lines[0].toLowerCase().indexOf("set")>=0?1:0;
  const cal=[],dep=[];
  for(let i=start;i<lines.length;i++){
    const p=lines[i].split(/[,;\t]/).map(function(t){return t.trim();});
    if(p.length<2)continue;
    const tag=(p[0]||"").toLowerCase(),sc=parseFloat(p[1]);
    const er=(p.length>2&&p[2]!=="")?parseFloat(p[2]):null;
    if(!isFinite(sc))continue;
    (tag.indexOf("d")===0?dep:cal).push([sc,er]);
  }
  if(cal.length<30||dep.length<30){
    msg.textContent=fmt(T.m_small,{a:cal.length,b:dep.length});return;}
  if(cal.some(function(x){return x[1]===null||(x[1]!==0&&x[1]!==1);})){
    msg.textContent=T.m_calerr;return;}
  const bad=dep.some(function(x){return x[1]!==null&&x[1]!==0&&x[1]!==1;});
  if(bad){msg.textContent=T.m_calerr;return;}
  const nLab=dep.filter(function(x){return x[1]!==null;}).length;
  S.data={calScores:cal.map(function(x){return x[0];}),calErrors:cal.map(function(x){return x[1];}),
          depScores:dep.map(function(x){return x[0];}),
          depErrors:nLab>0?dep.map(function(x){return x[1];}):null};
  S.label=fmt(T.lab_mine,{a:cal.length,b:dep.length})+
          (nLab===0?T.lab_mine_nolab:(nLab===dep.length?T.lab_mine_lab:fmt(T.lab_mine_part,{n:nLab})))+
          (S.fromSample?T.lab_sample_suffix:"");
  msg.textContent="";
  budgetCards();show("s1");
}
function pct(x){return (100*x).toFixed(1)+"%";}
function run(){
  if(!S.data){
    document.getElementById("result").innerHTML=
      '<div class="verdict v-amber">'+T.v_logic_h+'<span>'+T.v_logic_s+'</span></div>';
    show("s3");document.getElementById("s4").classList.remove("hide");return;}
  const d=S.data,a=S.alpha,f=fitLambda(d.calScores,d.calErrors,a,0.10,25,120);
  const acc=d.depScores.map(function(s){return s<=f.lam;}),nAcc=acc.filter(Boolean).length,nDep=acc.length;
  let meas=null;
  if(d.depErrors){
    const idx=[]; for(let i=0;i<d.depErrors.length;i++) if(d.depErrors[i]!==null) idx.push(i);
    const nLab=idx.length;
    if(nLab>=5){
      const ls=idx.map(function(i){return d.depScores[i];}),
            le=idx.map(function(i){return d.depErrors[i];}),
            la=idx.map(function(i){return acc[i];});
      const accLab=la.filter(Boolean).length/nLab;
      const de=le.filter(function(_,j){return !la[j];}),ds=ls.filter(function(_,j){return !la[j];});
      const nd=de.length,ne=de.reduce(function(x,y){return x+y;},0);
      const rev=ds.map(function(v,i){return i;}).sort(function(i,j){return ds[i]-ds[j]||i-j;}).reverse();
      const q=Math.max(1,Math.round(0.25*nd)),cum=[];let c=0;
      rev.forEach(function(i){c+=de[i];cum.push(c);});
      const ol=oracleLambda(ls,le,a,25);
      const oAcc=isFinite(ol)?ls.filter(function(s){return s<=ol;}).length/nLab:0;
      meas={nLab:nLab,nMissing:nDep-nLab,accLab:accLab,deferRate:nd?ne/nd:null,
            c25:ne?cum[q-1]/ne:null,
            c50:ne?cum[Math.max(1,Math.round(0.5*nd))-1]/ne:null,
            head:nd?rev.slice(0,q).reduce(function(s,i){return s+de[i];},0)/q:null,
            tail:nd?rev.slice(-q).reduce(function(s,i){return s+de[i];},0)/q:null,
            oracle:oAcc};
      meas.ratio=oAcc>0?(accLab/oAcc):(accLab>0?Infinity:0);
    }
  }
  S.out={lam:f.lam,chosen:f.chosen,grid:f.grid,delta_m:f.delta_m,alpha:a,nCal:d.calScores.length,
         acceptance:nAcc/nDep,deferred:1-nAcc/nDep,label:S.label,meas:meas,
         noSolution:!isFinite(f.lam),reason:f.reason||null};
  render();
}
function nums(v){return v==null?"—":pct(v);}
function render(){
  const o=S.out,m=o.meas,R=o.alpha,B=PUB.budget[R],Q=PUB.queue[R],pRev=B[0],pErr=B[1];
  const c25=Q[0],c50=Q[1],ph=Q[2],pt=Q[3];
  let cls="v-green",head=T.v_ok_h,sub=fmt(T.v_ok_s,{a:pct(R)});
  if(o.noSolution){cls="v-red";head=T.v_none_h;sub=T.v_none_s;}
  else if(o.acceptance===1){cls="v-green";head=T.v_all_h;sub=T.v_all_s;}
  if(m&&m.ratio>1){cls="v-amber";head=T.v_warn_h;
    sub=fmt(T.v_warn_s,{r:isFinite(m.ratio)?m.ratio.toFixed(3):"∞"});}
  const row=function(lab,v,alt){return '<div><span class="lab">'+lab+'</span><span class="track"><span class="fill'+(alt?' alt':'')+'" style="width:'+Math.max(0,Math.min(100,100*v)).toFixed(1)+'%"></span></span><span class="pct">'+pct(v)+'</span></div>';};
  const ratioTxt=m&&m.ratio!=null?(isFinite(m.ratio)?m.ratio.toFixed(3):"∞"):"—";
  const note=(m?(m.nMissing>0?fmt(T.res_note_part,{n:m.nLab,m:m.nMissing}):T.res_note_lab):T.res_note_nolab)
             +" "+fmt(T.res_risk,{e:pErr.toFixed(3)});
  document.getElementById("result").innerHTML=
   '<div class="verdict '+cls+'">'+head+'<span>'+sub+'</span></div>'
  +'<div class="grid"><div class="kpi"><b>'+pct(o.acceptance)+'</b><span>'+T.k_accept+'</span></div>'
  +'<div class="kpi"><b>'+pct(o.deferred)+'</b><span>'+T.k_defer+'</span></div>'
  +'<div class="kpi"><b>'+(isFinite(o.lam)?o.lam.toFixed(4):"—")+'</b><span>'+T.k_lam+'</span></div>'
  +'<div class="kpi"><b>'+(m&&m.deferRate!=null?m.deferRate.toFixed(3):"—")+'</b><span>'+(m?T.k_drisk:T.k_drisk_nolab)+'</span></div></div>'
  +'<table><thead><tr><th>'+T.th_action+'</th><th>'+T.th_yours+'</th><th>'+T.th_paper+'</th></tr></thead><tbody>'
  +'<tr><td>'+T.r_review+'</td><td class="n">'+pct(o.deferred)+'</td><td class="n">'+pct(pRev)+'</td></tr>'
  +'<tr><td>'+T.r_c25+'</td><td class="n">'+(m?nums(m.c25):"—")+'</td><td class="n">'+pct(c25)+'</td></tr>'
  +'<tr><td>'+T.r_c50+'</td><td class="n">'+(m?nums(m.c50):"—")+'</td><td class="n">'+pct(c50)+'</td></tr>'
  +'<tr><td>'+T.r_head+'</td><td class="n">'+(m&&m.head!=null?m.head.toFixed(3):"—")+'</td><td class="n">'+ph.toFixed(3)+'</td></tr>'
  +'<tr><td>'+T.r_tail+'</td><td class="n">'+(m&&m.tail!=null?m.tail.toFixed(3):"—")+'</td><td class="n">'+pt.toFixed(3)+'</td></tr>'
  +'<tr><td>'+T.r_ratio+'</td><td class="n">'+ratioTxt+'</td><td class="n">0.744 / 0.766 / 3.05</td></tr></tbody></table>'
  +'<h3>'+T.q_h+'</h3><div class="curve">'
  +row(T.cur_c25,m&&m.c25!=null?m.c25:c25)
  +row(T.cur_c50,m&&m.c50!=null?m.c50:c50)
  +row(T.cur_k25,0.25,true)+row(T.cur_k50,0.50,true)+'</div>'
  +'<p class="muted">'+note+'</p>'
  +'<details><summary>'+T.res_read_h+'</summary><p>'+fmt(T.res_read,{a:pct(o.acceptance),b:pct(R),c:pct(o.deferred),d:(isFinite(o.lam)?o.lam.toFixed(4):"—")})+'</p></details>';
  show("s3");
  document.getElementById("s4").classList.remove("hide");
}
function conclusion(){
  const o=S.out;
  if(!o){toast(T.noplan);return;}
  const m=o.meas,lines=[T.c_title,
    T.c_data+(o.label||""),
    T.c_budget+pct(o.alpha)+T.c_budget_note,
    T.c_lam+(isFinite(o.lam)?o.lam.toFixed(4):T.c_none),
    T.c_accept+pct(o.acceptance)+T.c_defer+pct(o.deferred)];
  if(m&&m.deferRate!=null) lines.push(T.c_drisk+m.deferRate.toFixed(3));
  if(m&&m.ratio!=null) lines.push(T.c_ratio+(isFinite(m.ratio)?m.ratio.toFixed(3):"∞")+(m.ratio>1?T.c_ratio_bad:T.c_ratio_ok));
  lines.push(T.c_action);
  copyText(lines.join("\n"));
}
function copyText(t){
  const ok=function(){toast(T.m_toast_copy);};
  const fb=function(){
    const ta=document.createElement("textarea");ta.value=t;ta.style.position="fixed";ta.style.opacity="0";
    document.body.appendChild(ta);ta.select();
    let good=false;try{good=document.execCommand("copy");}catch(e){}
    document.body.removeChild(ta);
    if(good)ok();else window.prompt(T.m_copy_prompt,t);
  };
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t).then(ok,fb);}
  else fb();
}
function downloadJson(){
  if(!S.out){toast(T.m_toast_run);return;}
  const blob=new Blob([JSON.stringify(S.out,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="annotation_plan.json";a.click();
}
function restart(){S={data:null,alpha:0.10,out:null,label:"",fromSample:false};CUR="s0";budgetCards();show("s0");}
budgetCards();show("s0");
</script>
</body>
</html>
"""

def build_sample_csv():
    """A deterministic synthetic batch: 800 labelled calibration cells and 1200
    deployment cells of which 400 carry labels, so the audit ratio is computable
    and the remaining 800 show what a partly labelled file looks like."""
    rng = np.random.default_rng(SAMPLE_SEED)

    def draw(n):
        s = rng.beta(1.1, 1.0, size=n)
        p = np.clip(SAMPLE_C0 + SAMPLE_SEV * s ** SAMPLE_POW, 0.0, 1.0)
        return s, (rng.random(n) < p).astype(int)

    cal_s, cal_e = draw(SAMPLE_N_CAL)
    dep_s, dep_e = draw(SAMPLE_N_DEP)
    lab = np.zeros(SAMPLE_N_DEP, dtype=bool)
    lab[rng.choice(SAMPLE_N_DEP, size=SAMPLE_N_LAB, replace=False)] = True
    lines = ["set,score,error"]
    for s, e in zip(cal_s, cal_e):
        lines.append("cal,%.6f,%d" % (s, int(e)))
    for s, e, k in zip(dep_s, dep_e, lab):
        lines.append("dep,%.6f,%s" % (s, int(e) if k else ""))
    return "\n".join(lines) + "\n"


def load_demo_unit():
    """Exactly scripts/r59_adviser.py: load_demo(), so the page and the CLI agree."""
    if not os.path.exists(DEMO_UNIT):
        raise SystemExit("demo unit not found: %s" % DEMO_UNIT)
    d = np.load(DEMO_UNIT, allow_pickle=True)
    ci, ti = d["cal_rep"][:, 0], d["test_rep"][:, 0]
    return {"cal_scores": 1.0 - d["cal_probs"][ci].max(1),
            "cal_errors": (d["cal_probs"][ci].argmax(1) != d["cal_y"][ci]).astype(float),
            "dep_scores": 1.0 - d["test_probs"][ti].max(1),
            "dep_errors": (d["test_probs"][ti].argmax(1) != d["test_y"][ti]).astype(float)}


def _join_floats(values):
    """Full-precision floats as a comma-separated JavaScript array body."""
    return ",".join(repr(float(x)) for x in values)


def _join_digits(values):
    """0/1 error indicators as a comma-separated JavaScript array body."""
    return ",".join("1" if float(x) > 0.5 else "0" for x in values)


def main():
    only_zh, only_en = set(TEXT["zh"]) - set(TEXT["en"]), set(TEXT["en"]) - set(TEXT["zh"])
    assert not only_zh and not only_en, (
        "language tables differ -- zh only: %s ; en only: %s" % (sorted(only_zh), sorted(only_en)))

    unit = load_demo_unit()
    NCAL = int(unit["cal_scores"].size)
    NDEP = int(unit["dep_scores"].size)
    assert NCAL >= 30 and NDEP >= 30, "the demo unit is too small to be a demonstration"
    demo = ("{calScores:[" + _join_floats(unit["cal_scores"]) +
            "],calErrors:[" + _join_digits(unit["cal_errors"]) +
            "],depScores:[" + _join_floats(unit["dep_scores"]) +
            "],depErrors:[" + _join_digits(unit["dep_errors"]) + "]}")
    assert ";" not in demo, "the demo literal would not parse as JavaScript"

    sample = build_sample_csv()
    io.open(SAMPLE_OUT, "w", encoding="utf-8", newline="\n").write(sample)
    print("wrote", SAMPLE_OUT, len(sample), "bytes")

    for lang in ("zh", "en"):
        html = (SKELETON
                .replace("__LANG__", "zh-CN" if lang == "zh" else "en")
                .replace("__TITLE__", TEXT[lang]["meta_title"])
                .replace("__DEMO__", demo)
                .replace("__TEXT_JSON__", json.dumps(TEXT[lang], ensure_ascii=False, sort_keys=True))
                .replace("__SAMPLE_JSON__", json.dumps(sample, ensure_ascii=False)))
        io.open(OUT[lang], "w", encoding="utf-8", newline="\n").write(html)
        print("wrote", OUT[lang], len(open(OUT[lang], "rb").read()), "bytes")


if __name__ == "__main__":
    main()
