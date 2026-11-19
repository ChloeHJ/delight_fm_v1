suppressPackageStartupMessages({library(Seurat);library(Matrix);library(jsonlite)})
input<-Sys.getenv('ATLAS_MAMBA_INPUT_RDS')
out<-Sys.getenv('EXPORT_DIR')
donors<-read.csv(Sys.getenv('V12_DONORS_CSV'),colClasses=c(sample_id='character'),check.names=FALSE)
x<-readRDS(input)
excluded<-jsonlite::fromJSON(Sys.getenv('V12_EXCLUDED_AGES','[]'))
keep<-!as.character(x$AgeGroup) %in% excluded
x<-subset(x,cells=colnames(x)[keep])
stopifnot(!anyDuplicated(donors$sample_id),all(as.character(x$sample_id) %in% donors$sample_id))
idx<-match(as.character(x$sample_id),donors$sample_id)
x$SplitDonor<-donors$SplitDonor[idx]
# Apply the frozen donor-level age labels, including the retained colon <1-year assignments.
x$AgeGroup<-donors$AgeGroup[idx]
counts<-table(as.character(x$sample_id))
stopifnot(all(as.integer(counts[donors$sample_id])==donors$n_cells))
# Export the unchanged gene universe/HVG annotations and current RNA data layer.
# No temporary filtered RDS is required.
DefaultAssay(x)<-'RNA'
if(inherits(x[['RNA']],'Assay5'))x<-JoinLayers(x,assay='RNA')
if(!'data' %in% Layers(x[['RNA']]))x<-NormalizeData(x,assay='RNA',verbose=FALSE)
obs<-x@meta.data
stopifnot(all(c('BroadCellType','Cluster','Location','AgeGroup','PCW') %in% names(obs)))
obs$cell_id<-rownames(obs);obs$sample_id<-as.character(x$sample_id);obs$donor_id<-obs$sample_id
if(!'OriginalLineage' %in% names(obs) && 'Lineage' %in% names(obs))obs$OriginalLineage<-as.character(obs$Lineage)
obs$Lineage<-as.character(obs$BroadCellType);obs$location<-obs$Location
obs$PCW<-as.numeric(as.character(obs$PCW));stopifnot(all(is.finite(obs$PCW)&obs$PCW>0))
obs$log2_PCW<-log2(obs$PCW)
genes<-rownames(x[['RNA']]);expr<-GetAssayData(x,assay='RNA',layer='data')[genes,,drop=FALSE]
global<-x@misc$global_hvg;lineage<-unique(unlist(x@misc$lineage_hvg,use.names=FALSE))
if(!length(global)&&!length(lineage))global<-VariableFeatures(x)
stopifnot(length(global)+length(lineage)>0)
means<-Matrix::rowMeans(expr);sds<-sqrt(pmax(Matrix::rowMeans(expr^2)-means^2,1e-8))
var<-data.frame(gene=genes,mean_log_normalized=means,sd_log_normalized=sds,in_global_hvg=genes%in%global,in_any_lineage_hvg=genes%in%lineage)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
Matrix::writeMM(expr,file.path(out,'log_normalized_gene_by_cell.mtx'))
write.csv(var,file.path(out,'var.csv'),row.names=FALSE)
write.csv(obs,gzfile(file.path(out,'obs.csv.gz')),row.names=FALSE)
stopifnot('umap.harmony' %in% Reductions(x))
umap<-Embeddings(x,'umap.harmony')
write.csv(data.frame(cell_id=rownames(umap),umap_1=umap[,1],umap_2=umap[,2]),gzfile(file.path(out,'umap_harmony.csv.gz')),row.names=FALSE)
# Keep publication metadata portable across machines and dataset installations.
group<-Sys.getenv('ATLAS_MAMBA_LINEAGE_GROUP')
input_ref<-file.path('data','rna',basename(input))
donor_ref<-file.path('donors',paste0(group,'.csv'))
saveRDS(list(input=input_ref,donor_file=donor_ref,cells=ncol(x),genes=length(genes)),file.path(out,'export_provenance.rds'))
