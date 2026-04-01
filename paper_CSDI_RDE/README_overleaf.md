# Overleaf 使用说明

## 如何在 Overleaf 上使用本论文

### 步骤 1：创建新项目
1. 登录 Overleaf 账号
2. 点击 "New Project" → "Upload Project"
3. 选择 "Upload a Zip file" 选项

### 步骤 2：准备上传文件
将以下文件压缩成一个 zip 包：
- `paper_CSDI_RDE.tex` - 主论文文件
- `paper_CSDI_RDE_overleaf.tex` - 简化版本（可选）
- `figures/` 目录（如果有图表）

### 步骤 3：上传并编译
1. 上传 zip 包后，Overleaf 会自动解压
2. 打开 `paper_CSDI_RDE.tex` 文件
3. 点击 "Recompile" 按钮

### 常见问题及解决方案

#### 问题 1：中文显示问题
- **原因**：Overleaf 可能默认使用英文环境
- **解决方案**：确保文件开头包含 `\usepackage[UTF8]{ctex}`

#### 问题 2：图表路径问题
- **原因**：Overleaf 中的文件路径与本地不同
- **解决方案**：
  1. 在 Overleaf 中创建 `figures` 目录
  2. 上传所有图表文件到该目录
  3. 确保 `\includegraphics` 命令使用正确的路径

#### 问题 3：编译错误
- **原因**：可能是某些包的版本问题
- **解决方案**：
  - 尝试使用 `paper_CSDI_RDE_overleaf.tex` 简化版本
  - 检查是否有未闭合的环境或命令
  - 确保所有的 `\begin{...}` 都有对应的 `\end{...}`

### 注意事项

1. **Overleaf 环境**：Overleaf 使用 TeX Live 发行版，通常包含所有常用的包
2. **编译方式**：使用 PDFLaTeX 或 XeLaTeX 编译
3. **图表格式**：建议使用 PDF 或 PNG 格式的图表
4. **参考文献**：本论文使用标准的 `thebibliography` 环境，不需要额外的 `.bib` 文件

### 预览
如果需要快速预览效果，可以使用 `paper_CSDI_RDE_overleaf.tex` 文件，这是一个简化版本，包含了主要内容但减少了复杂度，更容易在 Overleaf 上编译通过。

## 文件说明

- `paper_CSDI_RDE.tex` - 完整的论文，包含所有章节和内容
- `paper_CSDI_RDE_overleaf.tex` - 简化版本，适合快速测试和预览
- `README_overleaf.md` - 本说明文件
- `references.bib` - BibTeX 格式的参考文献（备用）
- `figures/` - 图表目录（需要手动创建并添加图表）

## 联系方式

如有任何问题，请联系：[您的联系方式]
