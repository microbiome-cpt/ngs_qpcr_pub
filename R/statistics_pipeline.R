if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# List of required packages
packages <- c(
  "vegan", "dplyr", "tidyr", "zCompositions", "compositions", 
  "heplots", "MVN", "stringr", "ape", 
  "plotly", "ggplot2", "ggsci", "pals",
  "reshape2", "ggnewscale", "ggpubr", "ggdendro", 
  "patchwork", "htmlwidgets")

# Install all missing packages
install_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    tryCatch(
      install.packages(pkg, dependencies = TRUE),
      error = function(e) {
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
      }
    )
  }
}
invisible(sapply(packages, install_missing))

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))

set.seed(100)
setwd('C:/Users/IGrabarnik/Desktop/pcr_ngs/test_rep2') #set working directory

palette <- c(
  "#61A695", "#AA74B4", "#A51300", "#A0D9E9", "#5163F8",
  "#F7578F", "#14642F", "#1234FA", "#3471A1", '#FEE0D2',
  "#F2F0A0", "#FB6A4A", "#EF3D0A", "#DB181D", "#A51",
  "#6FD00D", "#7F400B", "#343008", "#580004", "#069600")


#CHD Dataset
df.rdp <- read.csv2('datasets/df.rdp.csv')  
df.silva <- read.csv2('datasets/df.silva.csv')
df.rdp.genus <- read.csv2('datasets/df.rdp.genus.csv')  
df.silva.genus <- read.csv2('datasets/df.silva.genus.csv')
df.pcr <- read.csv2('datasets/df.pcr.csv')

df.ngs.meta <- read.csv2('datasets/df.ngs.meta.csv')
df.pcr.meta <- read.csv2('datasets/df.pcr.meta.csv')

#CE Dataset
df.ce <- read.csv2('datasets/df.pcr.comp.csv')
df.comp <- df.ce[,-1]
df.comp.meta <- df.ce[,1:2]

rownames(df.rdp) <- df.ngs.meta$Name
rownames(df.silva) <- df.ngs.meta$Name
rownames(df.pcr) <- df.pcr.meta$Sample

### ECOSTAT PIPELINE ###
ecostat <- function(df.names) {
  
  for (df.name in df.names) {
    if (df.name!='df.pcr' & df.name!='df.comp') {
      
      outname <- paste0(df.name, '_statistics.txt')
      dataset <- get(df.name)
      sqrelt <- sqrt(dataset/rowSums(dataset))
      dist_matrix <- vegdist(sqrelt, method = "bray")
      df.meta <- df.ngs.meta
      sink(outname)
      
      #----PERMANOVA-----
      permanova <- adonis2(dist_matrix ~ Disease+Method+Age+BMI+Sex, 
                           data = df.meta, permutations=9999, by='margin')
      #----PERMDISP------
      bdDis <- permutest(betadisper(dist_matrix, df.meta$Disease, 
                                    bias.adjust = T), permutations=9999)
      bdMet <- permutest(betadisper(dist_matrix, df.meta$Method, 
                                    bias.adjust = T), permutation=9999)
      
      #----dbRDA---------
      cap <- capscale(dist_matrix ~ Disease+Method+Age+Sex+BMI, data = df.meta)
      caps <- anova.cca(cap, by='margin', model = 'full', permutations=9999)
      
      #----OUT-----------
      cat(df.name, '\n')
      cat('\n==== PERMANOVA =====')
      print(permanova)
      cat('\n==== PERMDISP =====')
      print(bdDis)
      print(bdMet)
      cat('\n==== CAP =====')
      print(cap)
      print(caps)
      sink()
    } 
    
    else {
      outname <- paste0(df.name, '_statistics.txt')
      dataset <- get(df.name)
      
      df.pcr_rel <- as.matrix((dataset[,-1])/dataset[,1])
      df_zero_fixed <- cmultRepl(df.pcr_rel, method = "CZM", label = 0)  #multiplicative zero replacement
      clr_matrix <- clr(df_zero_fixed)
      dist_pcr <- dist(clr_matrix, method = "euclidean") #Aitchison distance
      
      if (df.name=='df.pcr') {
      sink(outname)
      #----MANCOVA-------
      mancova_model <- manova(ilr(df_zero_fixed) ~ Method+Disease+Age+Sex+BMI, data = df.pcr.meta)
      mancova.sum <- summary(mancova_model, test = "Pillai")  
      #summary.aov(mancova_model) 
      
      #--assumptions_test
      res <- residuals(mancova_model)
      mardia <- MVN::mvn(as.data.frame(res), mvn_test = "mardia")$multivariate_normality
      boxMet <- heplots::boxM(as.matrix(res), df.pcr.meta$Method)
      boxDis <- heplots::boxM(as.matrix(res), df.pcr.meta$Disease)
      
      #----PERMANOVA-----
      pcr_permanova <-  adonis2(dist_pcr ~ Method+Disease+Sex+Age+BMI,
                                data = df.pcr.meta, permutations = 9999, by='margin')
      
      #----PERMDISP------
      bdMet <- permutest(betadisper(dist_pcr, df.pcr.meta$Method, 
                                    bias.adjust = T), permutation=9999)
      bdDis <- permutest(betadisper(dist_pcr, df.pcr.meta$Disease, 
                                    bias.adjust = T), permutations=9999)
      
      #----dbRDA---------
      cap <- capscale(dist_pcr ~ Disease+Method+Age+Sex+BMI, data = df.pcr.meta)
      caps <- anova.cca(cap, by='margin', model = 'full', permutations=9999)
      
      #----OUT-----------
      cat(df.name, '\n')
      cat('==== MANCOVA =====')
      print(mancova.sum)
      cat('\n  Multivariate Normality Tests  \n')
      print(mardia)
      print(boxMet)
      print(boxDis)
      cat('\n==== PERMANOVA =====')
      print(pcr_permanova)
      cat('\n==== PERMDISP =====')
      print(bdMet)
      print(bdDis)
      cat('\n==== CAP =====')
      print(cap)
      print(caps)

      } else if (df.name=='df.comp') {
        ce_permanova <-  adonis2(dist_pcr ~ KitType, data = df.comp.meta, 
                                 permutations = 9999, by='margin')
        cat('\n==== CE DATASET PERMANOVA =====\n')
        print(ce_permanova)
        sink()
      }
    } 
  }
}


### PICTURES PIPELINE
ecostat_plots <- function(df.names) {

  df.meta.merged <- df.meta.new %>%  filter(Disease %in% c("(cond_healthy)", "(HFpEF)")) %>%
    mutate(Disease = "(Merged)") %>% bind_rows(df.meta.new)
  
  ngs_age_dis <- ggplot(data=df.meta.merged, aes(x = factor(Disease, levels = dis_order), y= Age, fill=Disease))+
    geom_boxplot() + theme_pubclean()+xlab('')+scale_fill_npg(alpha = 0.5)+ theme(axis.text.x = element_text(angle = 25, hjust=1))
  ngs_bmi_dis <- ggplot(data=df.meta.merged, aes(x = factor(Disease, levels = dis_order), y= BMI, fill=Disease))+
    geom_boxplot() + theme_pubclean()+xlab('')+scale_fill_npg(alpha = 0.5)+ theme(axis.text.x = element_text(angle = 25, hjust =1))
  ngs_age_met <- ggplot(data=df.meta.merged, aes(x = Method, y= Age, fill=Method))+geom_boxplot()+theme_pubclean()+xlab('')+
    scale_fill_manual(values = c("#6F99ADFF", "#E18A27FF"))
  ngs_bmi_met <- ggplot(data=df.meta.merged, aes(x = Method, y= BMI, fill=Method))+geom_boxplot()+theme_pubclean()+xlab('')+
    scale_fill_manual(values = c("#6F99ADFF", "#E18A27FF"))
  
  df.meta.merged <- df.pcr.meta %>%  filter(Disease %in% c("(cond_healthy)", "(HFpEF)")) %>%
    mutate(Disease = "(Merged)") %>% bind_rows(df.pcr.meta)
  
  pcr_age_dis <- ggplot(data=df.meta.merged, aes(x = factor(Disease, levels = dis_order), y= Age, fill=Disease))+
    geom_boxplot() + theme_pubclean()+xlab('')+scale_fill_npg(alpha = 0.5)+ theme(axis.text.x = element_text(angle = 25, hjust=1))
  pcr_bmi_dis <- ggplot(data=df.meta.merged, aes(x = factor(Disease, levels = dis_order), y= BMI, fill=Disease))+
    geom_boxplot()+ theme_pubclean()+xlab('')+scale_fill_npg(alpha = 0.5) + theme(axis.text.x = element_text(angle = 25, hjust=1))
  pcr_age_met <- ggplot(data=df.meta.merged, aes(x = Method, y= Age, fill=Method))+geom_boxplot()+theme_pubclean()+xlab('')+
    scale_fill_manual(values = c("#6F99ADFF", "#E18A27FF"))
  pcr_bmi_met <- ggplot(data=df.meta.merged, aes(x = Method, y= BMI, fill=Method))+geom_boxplot()+theme_pubclean()+xlab('')+
    scale_fill_manual(values = c("#6F99ADFF", "#E18A27FF"))
  
  (pcr_age_met | ngs_age_met | pcr_bmi_met | pcr_bmi_met) / (pcr_age_dis | ngs_age_dis | pcr_bmi_dis | ngs_bmi_dis)+
    plot_layout(widths = c(3,1), guides = 'collect', axes = 'collect')+
    plot_annotation(tag_levels = 'A', theme = theme(legend.position = 'bottom'))
  ggsave('disease_groups_characteristics.png', path='plots', dpi=300, height = 8, width = 11)
  
  
  
  for (df.name in df.names) {
    outname <- paste0(df.name, '_fig.png')
    if (df.name!='df.pcr' & df.name!='df.comp') {
      dataset <- get(df.name)
      dummy <- dataset
      sqrelt <- sqrt(dummy/rowSums(dummy))
      dist_matrix <- vegdist(sqrelt, method = "bray")
      df.meta <- df.ngs.meta
    } else {
      dataset <- get(df.name)
      df.pcr_rel <- as.matrix((dataset[,-1])/dataset[,1])
      df_zero_fixed <- cmultRepl(df.pcr_rel, method = "CZM", label = 0)  #multiplicative zero replacement
      clr_matrix <- clr(df_zero_fixed)
      dist_matrix <- dist(clr_matrix, method = "euclidean")
      df.meta <- df.pcr.meta
    }
    if (df.name!='df.comp') {
    
    #---PCOA----   
    pcoa_result <- pcoa(dist_matrix)
    pcoa_df <- as.data.frame(pcoa_result$vectors[, 1:2])  
    pcoa_df$Disease <- df.meta$Disease
    pcoa_df$Method  <- df.meta$Method
    
    #Centroids
    centroids_d <- pcoa_df %>%
      group_by(Disease) %>%
      summarise(PC1 = mean(Axis.1), PC2 = mean(Axis.2))
    
    centroids_m <- pcoa_df %>%
      group_by(Method) %>%
      summarise(PC1 = mean(Axis.1), PC2 = mean(Axis.2))
    
    pcoa_df$Disease <- factor(pcoa_df$Disease, levels = c("(HFrEF)", "(HFpEF)", "(cond_healthy)", "(CAD)"))
    pcoa_dis <- ggplot(pcoa_df, aes(x = Axis.1, y = Axis.2, color = Disease)) +
      geom_point(alpha = 0.8, size=3, show.legend = F) +
      stat_ellipse(type = "t", linetype = 2, show.legend = FALSE) +  #ellipses
      geom_point(data = centroids_d, aes(x = PC1, y = PC2), size = 4, shape = 3, show.legend = FALSE) +
      geom_vline(aes(xintercept=0), linetype=3, color = 'grey')+ #, alpha=0.8) +
      geom_hline(aes(yintercept=0), linetype=3, color = 'grey')+ #, alpha=0.8) +
      scale_color_npg(breaks = c("(CAD)", "(HFrEF)", "(HFpEF)", "(cond_healthy)"),
                      labels = c("CAD", "HFrEF", "HFpEF", "Control"))+
      labs(x = "PCoA1", y = "", color='Disease Status')+
      theme_bw(base_size = 18) +
      theme(panel.grid.major = element_line(colour = "white"), 
            panel.grid.minor = element_line(colour = "white"),
            panel.border = element_rect(linewidth = 0.7))
    
    pcoa_kit <- ggplot(pcoa_df, aes(x = Axis.1, y = Axis.2, color = Method)) +
      geom_point(alpha = 0.8, size=3) +
      stat_ellipse(type = "t", linetype = 2, show.legend = FALSE) +  
      geom_point(data = centroids_m, aes(x = PC1, y = PC2), size = 4, shape = 3, show.legend = FALSE) +
      scale_color_manual(values = c("#6F99ADFF", "#E18A27FF"))+
      guides(color = guide_legend(override.aes = list(size = 4)))+
      geom_vline(aes(xintercept=0), linetype=3, color = 'grey')+ #, alpha=0.8) +
      geom_hline(aes(yintercept=0), linetype=3, color = 'grey')+ #, alpha=0.8) +
      labs(x = "PCoA1", y = "PCoA2", color='Kit Color')+
      theme_bw(base_size = 18)+
      theme(panel.grid.major = element_line(colour = "white"), 
            panel.grid.minor = element_line(colour = "white"),
            panel.border = element_rect(linewidth = 0.7))
    
    #---CAPSCALE (dbRDA)----
    cap <- capscale(dist_matrix ~ Disease+Method+Age+Sex+BMI, data = df.meta)
    caps <- anova.cca(cap, by='margin', model = 'full', permutations=9999)
    caps_round <- cbind(round(caps[1:5,1:3],2), caps[1:5,4])
    colnames(caps_round)[4] <- 'Pr(>F)'
    for (i in 1:nrow(caps_round)) {
      if (caps_round[i,4] <= 0.001) {caps_round[i,4] <- paste0(caps_round[i,4], '***')}
      else if (caps_round[i,4] <= 0.01) {caps_round[i,4] <- paste0(caps_round[i,4], '**')}
      else if (caps_round[i,4] <= 0.05) {caps_round[i,4] <- paste0(caps_round[i,4], '*')}
      else if (is.character(caps_round[i,4])) {caps_round[i,4] <- paste0(caps_round[i,4], '***')}
    }
    
    df.meta$Disease <- factor(df.meta$Disease, levels = c("(HFrEF)", "(HFpEF)", "(cond_healthy)", "(CAD)"))
    caplot <- ggplot(scores(cap)$constraints, aes(x=CAP1, y=CAP2))+
      geom_point(aes(color=df.meta$Disease, shape=df.meta$Method), size=4, alpha=0.9)+ 
      geom_hline(yintercept = 0, linetype = 'dotted')+
      geom_vline(xintercept = 0, linetype = 'dotted')+ 
      scale_color_npg(breaks = c("(CAD)", "(HFrEF)", "(HFpEF)", "(cond_healthy)"),
                      labels = c("CAD", "HFrEF", "HFpEF", "Control"))+ 
      geom_segment(data=scores(cap)$biplot[c(5,7),], aes(x=0, xend=CAP1, y=0, yend=CAP2), 
                   arrow = arrow(length = unit(0.25, "cm"),angle = 18), color='blue')+
      geom_text(data=scores(cap)$biplot[c(5,7),], aes(x=CAP1, y=CAP2, 
                                                      label=rownames(scores(cap)$biplot)[c(5,7)]), color='blue', size=4, hjust=1, vjust=1)+
      labs(color='Disease Status', shape='Kit Shape')+
      xlab(paste0('dbRDA1 (', round(cap$CCA$eig[1]/cap$tot.chi, digits = 2), '%)'))+
      ylab(paste0('dbRDA2 (', round(cap$CCA$eig[2]/cap$tot.chi, digits = 2), '%)'))+
      guides(shape = guide_legend(override.aes = list(size = 4)))+
      theme_bw(base_size = 18)
    
    #tab <- tableGrob(caps_round, theme=grid_theme) %>% 
    #  tab_add_hline(at.row = 1)
    #caplotable <- ggarrange(caplot, tab, nrow=2, heights = c(4, 1))
    
    plots <- (pcoa_kit + pcoa_dis) / caplot + 
      plot_layout(heights = c(1, 1), guides = 'collect', axes = 'collect')+
      plot_annotation(tag_levels = 'A', theme = theme(legend.position = 'bottom'))+
      guides(shape = guide_legend(order = 2))
    
    ggsave(outname, plot = plots,  path = 'plots', dpi=300, bg='white', width = 10, height = 11.5)
    }
    
    if (df.name!='df.pcr' & df.name!='df.comp') {
      #---NMDS----
      nmds <- metaMDS(dummy, autotransform = T, distance = 'bray', 
                      try=500, trymax=1000, k=3)
      nmds.scores <- as.data.frame(scores(nmds, display = 'sites'))
      nmds.scores$meth <- df.meta$Method
      nmds.scores$diag <- df.meta$Disease
      nmds.scores$name <- df.meta$Name
      nmds.scores$sum <- rowSums(dummy)
      for (i in 1:nrow(dummy)) {
        nmds.scores$dom[i] <- names(sort(as.matrix(dummy)[i,], decreasing = T)[1])
      }
      nmds.scores$dom <- sub('f_', '', nmds.scores$dom) #taxon prefix removal
      nmds.scores$dom <- sub('o_', '', nmds.scores$dom) #taxon prefix removal
      nmds.scores$age <- df.meta$Age
      nmds.scores$sex <- df.meta$Sex
      nmds.scores$bmi <- df.meta$BMI
      
      #Loading vectors
      env.fit <- envfit(nmds, dummy, permutations=999, choices=c(1,2,3))
      significant <- env.fit$vectors$pvals < 0.05 #p-value filtering...
      if (any(significant)) {
        env.fit$vectors$arrows <- env.fit$vectors$arrows[significant, , drop = FALSE]
        env.fit$vectors$r <- env.fit$vectors$r[significant]
        env.fit$vectors$pvals <- env.fit$vectors$pvals[significant]
      } else {
        env.fit$vectors <- NULL
      }        
      vectors <- as.data.frame(scores(env.fit, display = "vectors"))
      vectors$Taxon <- rownames(vectors)
      vectors3 <- vectors[names(colSums(dummy))[1:15],] # <- top 15 most abundant taxa vectors filtering
      vectors3$Taxon <-  sub('f_', '', vectors3$Taxon)  #taxon prefix removal
      
      #----huge plotly 3d NMDS plot code with loading vectors
      # if(1==1) is for easy code collapse :)
      if (1==1) {
        # Create 3d plot
        fig <- plot_ly(
          data = nmds.scores,
          x = ~NMDS1,
          y = ~NMDS2,
          z = ~NMDS3,
          type = "scatter3d",
          mode = "markers",
          color = ~diag,
          text = ~name,
          marker = list(size = 7),
          colors = ggsci::pal_npg()(4)
          ) %>% layout(showlegend=TRUE, legend = list(font = list(size = 21),
                                                      itemsizing = 'constant',
                                                      marker = list(size = 7)))
        
        # Add vectors
        for (i in 1:nrow(vectors3)) {
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "lines",
            x = c(0, vectors3$NMDS1[i]),
            y = c(0, vectors3$NMDS2[i]),
            z = c(0, vectors3$NMDS3[i]),
            line = list(color = 'black', width = 3),
            showlegend = FALSE, inherit = F
          )
          
          # Sign vectors
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "text",
            x = vectors3$NMDS1[i],
            y = vectors3$NMDS2[i],
            z = vectors3$NMDS3[i],
            text = vectors3$Taxon[i],
            textfont = list(size = 17),
            showlegend = FALSE, inherit = F
          )
        }
        
        # Save final plot
        saveWidget(fig, file = paste0(df.name, '_nmds_3d_disease.html'), selfcontained = TRUE)
      } #disease plot
      if (1==1) {
        # Create 3d plot
        fig <- plot_ly(
          data = nmds.scores,
          x = ~NMDS1,
          y = ~NMDS2,
          z = ~NMDS3,
          type = "scatter3d",
          mode = "markers",
          color = ~meth,
          text = ~name,
          marker = list(size = 7),
          colors = c("#6F99ADFF", "#E18A27FF")
        ) %>% layout(showlegend=TRUE, legend = list(font = list(size = 21),
                                                    itemsizing = 'constant',
                                                    marker = list(size = 7)))
        
        # Add vectors
        for (i in 1:nrow(vectors3)) {
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "lines",
            x = c(0, vectors3$NMDS1[i]),
            y = c(0, vectors3$NMDS2[i]),
            z = c(0, vectors3$NMDS3[i]),
            line = list(color = 'black', width = 3),
            showlegend = FALSE, inherit = F
          )
          
          # Sign vectors
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "text",
            x = vectors3$NMDS1[i],
            y = vectors3$NMDS2[i],
            z = vectors3$NMDS3[i],
            text = vectors3$Taxon[i],
            textfont = list(size = 17),
            showlegend = FALSE, inherit = F
          )
        }
        
        # Save final plot
        saveWidget(fig, file = paste0(df.name, '_nmds_3d_method.html'), selfcontained = TRUE)
      } #method plot
      if (1==1) {
        # Create 3d plot
        fig <- plot_ly(
          data = nmds.scores,
          x = ~NMDS1,
          y = ~NMDS2,
          z = ~NMDS3,
          type = "scatter3d",
          mode = "markers",
          color = ~dom,
          text = ~name,
          marker = list(size = 7),
          colors = pals::tol()
        ) %>% layout(showlegend=TRUE, legend = list(font = list(size = 21),
                                                                itemsizing = 'constant',
                                                                marker = list(size = 7)))
        
        # Add vectors
        for (i in 1:nrow(vectors3)) {
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "lines",
            x = c(0, vectors3$NMDS1[i]),
            y = c(0, vectors3$NMDS2[i]),
            z = c(0, vectors3$NMDS3[i]),
            line = list(color = 'black', width = 3),
            showlegend = FALSE, inherit = F
          )
          
          # Sign vectors
          fig <- fig %>% add_trace(
            type = "scatter3d",
            mode = "text",
            x = vectors3$NMDS1[i],
            y = vectors3$NMDS2[i],
            z = vectors3$NMDS3[i],
            text = vectors3$Taxon[i],
            textfont = list(size = 17),
            showlegend = FALSE, inherit = F
          )
        }
        
        # Save final plot
        saveWidget(fig, file = paste0(df.name, '_nmds_3d_dominants.html'), selfcontained = TRUE)
      } #dominant taxa plot
      
      #----NMDS with anthropometric factors annotation (Fig. S2)
      nmds_age <- ggplot(nmds.scores, aes(x=NMDS1, y=NMDS2, color = age, shape=sex))+
        geom_point(size = 2, alpha = 0.85)+
        scale_colour_viridis_c()+
        theme_minimal()+labs(color='Age', shape='Sex')+
        theme(legend.position = 'right')
      nmds_bmi <- ggplot(nmds.scores, aes(x=NMDS1, y=NMDS2, color = bmi, shape=sex))+
        geom_point(size = 2, alpha = 0.85)+
        theme_minimal()+labs(color='BMI', shape='Sex')+
        theme(legend.position = 'right')
      ggarrange(nmds_age, nmds_bmi, nrow = 2)
      ggsave(paste0(df.name, '_nmds_clinic.png'), path='plots', dpi=300, bg='white',
             width = 5, height = 7)
      
      
      #----Relative abundance barplots (microbiome profiles) (Fig. S4)
      rownames(dummy) <- nmds.scores$name
      colnames(dummy) <- sapply(strsplit(colnames(dummy), "_"), `[`, 2)
      
      # Clustering
      hclust_res <- hclust(vegdist(sqrelt, method = "bray"), method = "average")  
      sample_order <- hclust_res$labels[hclust_res$order]
      
      rel_dummy <- as.data.frame(dummy / rowSums(dummy))
      top_taxa <- names(sort(colSums(rel_dummy), decreasing = T)[1:20])
      bar_dummy <- melt(cbind(Sample = rownames(dummy), rel_dummy[, top_taxa], 
                              dom = nmds.scores$dom,
                              diag = nmds.scores$diag,
                              meth = nmds.scores$meth), 
                        id.vars = c("Sample", 'dom', 'diag', 'meth'), 
                        variable.name = "Taxon", 
                        value.name = "Abundance")
      bar_dummy$Sample <- factor(bar_dummy$Sample, levels = sample_order)
      
      barplots <- ggplot(bar_dummy, aes(x = Abundance, y = Sample, fill = Taxon)) +
        geom_bar(stat = "identity") +
        scale_fill_manual(values = palette)+
        new_scale_fill()+
        geom_text(aes(label=dom, x = 1), check_overlap = F, size = 2.8, hjust = 0) +
        geom_tile(aes(x=-0.02, fill = meth), width = 0.02, height = 1)+
        scale_fill_manual(values = c("#6F99ADFF", "#E18A27FF"))+
        labs(fill='KitType')+
        new_scale_fill()+
        geom_tile(aes(x=-0.04, fill = diag), width = 0.02, height = 1)+
        scale_fill_npg()+
        labs(fill='Disease')+
        theme_pubclean() +
        theme(#axis.text.x = element_text(angle = 90, vjust=0.5),
              axis.title.y = element_blank(),   # remove y axis title
              axis.title.x = element_blank(),   # remove x axis title
              axis.text.y = element_text(margin = margin(r = 0), vjust = 0.3),
              axis.ticks.y = element_blank(),
              panel.grid.minor = element_blank(),
              #axis.ticks.x = element_blank(),
              legend.position = 'right',
              legend.box = 'horizontal',
              legend.box.just = 'left')+
        scale_y_discrete(expand = c(0, 0))+guides(fill=guide_legend(order=2, ncol=2))
      
      dend <- as.dendrogram(hclust(vegdist(sqrelt, method = "bray"), method = "average"))
      #dend <- set(dend, "labels", '')
      dend_plot <- ggdendrogram(dend, rotate = TRUE, size=2, leaf_labels = F)+
        scale_y_reverse(expand = expansion(mult=c(.1,0)))+
        theme_void()+
        theme(plot.margin = margin(r = 0), panel.grid = element_blank())+
        scale_x_continuous(expand = expansion(mult=c(.018,.0084)))
      
      ggarrange(dend_plot, barplots, widths = c(1, 8),
                common.legend = T, legend = 'bottom')
      ggsave(paste0(df.name, '_barplot_fig.png'), path='plots', dpi=300, bg='white', 
             height = 22, width = 17)
    } 
    
    #----Extraction kit shifts plots (Fig. 4)
    else if (df.name=='df.pcr') {
      pcoa_chd <- pcoa_df
      singles <- pcoa_chd[1:123,] #first 123 rows are once-extracted samples
      doubles <- pcoa_chd[124:197,]
      fsd_chd <- doubles %>% filter(Method=='FS')
      pfd_chd <- doubles %>% filter(Method=='PF')
    } else if (df.name=='df.comp') {
      df.meta <- df.comp.meta
      pcoa_result <- pcoa(dist_matrix)
      pcoa_df <- as.data.frame(pcoa_result$vectors[, 1:2])  
      pcoa_df$Method  <- df.meta$KitType
      pcoa_ce <- pcoa_df
      fsd_ce <- pcoa_ce %>% filter(Method=='Fast DNA Stool')
      pfd_ce <- pcoa_ce %>% filter(Method=='PowerFecal')
    }
  }
    shift_chd <- ggplot(doubles, aes(x = Axis.1, y = Axis.2, color = Method)) +
    geom_segment(data=pfd_chd, aes(x=Axis.1, xend=fsd_chd$Axis.1, y=Axis.2, yend=fsd_chd$Axis.2), 
                 color='black', linetype = 2, alpha=0.5, linewidth = 0.8, inherit.aes = F) +
    geom_point(alpha = 0.9, size=3) +
    geom_point(data=singles, aes(x = Axis.1, y = Axis.2, color = Method), alpha=0.2, size=4) +
    scale_color_manual(values = c("#6F99ADFF", "#E18A27FF")) +
    guides(color = guide_legend(override.aes = list(size = 4))) +
    geom_vline(aes(xintercept=0), linetype=3, color = 'grey', alpha=0.8) +
    geom_hline(aes(yintercept=0), linetype=3, color = 'grey', alpha=0.8) +
#    labs(x = "PCoA1", y = "PCoA2", color='Kit Color')+
    labs(y="", x="", color='Kit Color')+
      theme_bw(base_size = 18) +
      theme(panel.grid.major = element_line(colour = "white"), 
            panel.grid.minor = element_line(colour = "white"),
            panel.border = element_rect(linewidth = 0.7))

  shift_ce <- ggplot(pcoa_ce, aes(x = Axis.1, y = Axis.2, color = Method)) +
    geom_segment(data=pfd_ce, aes(x=Axis.1, xend=fsd_ce$Axis.1, y=Axis.2, yend=fsd_ce$Axis.2), 
                 color='black', linetype = 2, alpha=0.5, linewidth = 0.8, inherit.aes = F) +
    geom_point(alpha = 0.85, size=3) +
    scale_color_manual(values = c("#8F99ADFF", "#D19A27FF"))+
    guides(color = guide_legend(override.aes = list(size = 4)))+
    geom_vline(aes(xintercept=0), linetype=3, color = 'grey', alpha=0.8) +
    geom_hline(aes(yintercept=0), linetype=3, color = 'grey', alpha=0.8) +
    labs(x = "PCoA1", y = "PCoA2", color='Kit Color')+
    theme_bw(base_size = 18) +
    theme(panel.grid.major = element_line(colour = "white"), 
          panel.grid.minor = element_line(colour = "white"),
          panel.border = element_rect(linewidth = 0.7))
  


pdes <- "
AB
CC
DE
"
   (pcoa_kit + pcoa_dis + caplot + shift_chd + shift_ce)  +
    plot_layout(design = pdes, guides = 'collect')+
    plot_annotation(tag_levels = 'A', theme = theme(legend.position = 'bottom'))+
    guides(shape = guide_legend(order = 2))
  
  ggsave('df.pcr_method_shift_fig2.png', path='plots', dpi=300, bg='white', width = 10, height = 16.5)

}


#start the pipeline on the list of selected datasets:
df.names <- c('df.rdp', 'df.silva', 'df.comp', 'df.pcr')
ecostat(df.names)
ecostat_plots(df.names)


#---Datasets and reference databases comparison----
spearman_table <- function() {
rownames(df.silva) <- paste0(df.ngs.meta$Name, '-', df.ngs.meta$Method)
rownames(df.rdp) <- rownames(df.silva)
rownames(df.rdp.genus) <- rownames(df.silva)
rownames(df.silva.genus) <- rownames(df.silva)
idx <- intersect(rownames(df.pcr), rownames(df.rdp))

pcr_systems <- gsub('PCR_|.sp.|.faecalis|_prausnitzii', '', colnames(df.pcr))[-1]
pcr <- df.pcr[,-1]/df.pcr[,1]


rdp_systems_g <- df.rdp.genus %>%
  select(matches(paste('g_', pcr_systems, sep='', collapse = "|")))
rdp_systems_f <- df.rdp %>%
  select(matches(paste('f_', pcr_systems, sep='', collapse = "|")))
rdp_systems <- cbind(rdp_systems_f[,-3]/rowSums(df.rdp), 
                     rdp_systems_g/rowSums(df.rdp.genus))

silva_systems_g <- df.silva.genus %>%
  select(matches(paste('g_', pcr_systems, sep='', collapse = "|")))
silva_systems_f <- df.silva %>%
  select(matches(paste('f_', pcr_systems, sep='', collapse = "|")))
silva_systems <- cbind(silva_systems_f/rowSums(df.silva), 
                       silva_systems_g[,-4]/rowSums(df.silva.genus))

ord <- unlist(lapply(pcr_systems, function(p) grep(p, colnames(rdp_systems))))
rdp_systems <- rdp_systems[,ord]

ord <- unlist(lapply(pcr_systems, function(p) grep(p, colnames(silva_systems))))
silva_systems <- silva_systems[,ord]



cor_silva <- NULL
cor_rdp   <- NULL
cor_bases <- NULL
for (i in 1:11) {
  cor_silva <- c(cor_silva, cor(pcr[idx,i], silva_systems[idx,i], method='spearman'))
  cor_rdp   <- c(cor_rdp, cor(pcr[idx,i], rdp_systems[idx,i], method='spearman'))
  cor_bases <- c(cor_bases, cor(rdp_systems[,i], silva_systems[,i], method='spearman'))
}
names(cor_silva) <- pcr_systems
names(cor_rdp) <- pcr_systems
names(cor_bases) <- pcr_systems

pval_silva <- NULL
pval_rdp   <- NULL
pval_bases <- NULL
for (i in 1:11) {
  pval_silva <- c(pval_silva, cor.test(pcr[idx,i], silva_systems[idx,i], method='spearman')$p.value)
  pval_rdp   <- c(pval_rdp,   cor.test(pcr[idx,i], rdp_systems[idx,i], method='spearman')$p.value)
  pval_bases <- c(pval_bases, cor.test(rdp_systems[,i], silva_systems[,i], method='spearman')$p.value)
}
pval_silva  <- p.adjust(pval_silva, method = "BH")
pval_rdp    <- p.adjust(pval_rdp,   method = "BH")
pval_bases  <- p.adjust(pval_bases, method = "BH")
names(pval_silva) <- pcr_systems
names(pval_rdp) <- pcr_systems
names(pval_bases) <- pcr_systems
sink('spearman.txt')
cat('SPEARMAN CORRELATION COEFFICIENT\n')
print(as.data.frame(cbind(cor_silva, cor_rdp, cor_bases)))
cat('\nP-VALUES (FDR-CORRECTED)\n')
print(as.data.frame(cbind(pval_silva, pval_rdp, pval_bases)))
sink()

#------Spearman Correlation Scatter Plots (Fig. S3)-------
all_data <- list()

for (i in 1:11) {
  data_silva <- data.frame(
    X = df.pcr[idx, i+1],
    Y = silva_systems[idx, i] * df.pcr[idx, 1],
    Method = "SILVA",
    Taxon = colnames(silva_systems)[i]
  )
  
  data_rdp <- data.frame(
    X = df.pcr[idx, i+1],
    Y = rdp_systems[idx, i] * df.pcr[idx, 1],
    Method = "RDP",
    Taxon = colnames(rdp_systems)[i]
  )
  
  all_data[[i]] <- rbind(data_silva, data_rdp)
}

plot_data <- do.call(rbind, all_data)

p <- ggplot(plot_data, aes(x = X, y = Y, color = Method)) +
  geom_smooth(method = 'lm', alpha = 0.3, linewidth = 0.5)+geom_point(alpha = 0.5) + 
  stat_cor(aes(color = Method),
           method = "spearman",
           label.x.npc = "left",
           label.y.npc = "top",
           cor.coef.name = "rho",
           show.legend = FALSE) +
  scale_color_manual(values = c("black", "red"))+
  xlab("qPCR") +
  ylab("NGS") +
  facet_wrap(~ Taxon, scales = "free") +
  theme_bw()+theme(legend.position = 'bottom')

ggsave("spearman_scatters_faceted.png",
       plot = p,
       path = "plots",
       dpi = 300,
       bg = "white",
       height = 8, width = 12)
}
spearman_table()

#check your current working directory for the results